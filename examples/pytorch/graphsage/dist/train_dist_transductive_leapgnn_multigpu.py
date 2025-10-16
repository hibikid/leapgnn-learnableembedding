import argparse
import time

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
from dgl.distributed import DistEmbedding
from train_dist import DistSAGE, compute_acc
from local_emboptimizer import SparseAdam
import os

#----------leapgnn-start-----------

"""
lessjp 功能实现思路:
1. 前k个epoch 构造矩阵，n*n, 每个格子填写 miss_rate；
2. 用启发式方法调整每个 iteration 的计算矩阵图；保证每个iteration、每个模型的移动的次数都是相同的；
   每次迭代后，根据生成的计算矩阵图，构建模型移动路径链，按照这个链进行send_recv；
3. 一次迭代后，总计k个epoch 的运行时间，直到平均epoch运行时间小于等于目标时间或者移动次数为0，算法迭代停止；
"""

def reverse_columns(arr, k):
    """
    把第k列的数据轮转到第0列
    """
    num_cols = arr.shape[1]
    reversed_arr = np.concatenate((arr[:, k:num_cols], arr[:, 0:k]), axis=1)
    return reversed_arr

def substract_columns(arr):
    """
    根据 offsets arr 计算出每个 sub_batch 包含的点的数量
    """
    prev_cols = arr[ :, :-1]
    next_cols = arr[ :, 1:]
    return next_cols - prev_cols

def get_model_trace(jp_times, world_size, sub_batch_offsets):
    # 从各个节点 gather 每个 micro-batch 的样本数量
    sub_batches = [[None for _ in range(len(sub_batch_offsets))] for _ in range(world_size)] # world_size * len_of_sub_batches_of_each_rank
    # print('sub_batches', sub_batches)
    th.distributed.all_gather_object(sub_batches, sub_batch_offsets)
    # convert into 2-d numpy array
    sub_batches = np.array(sub_batches)
    sub_batches_size = substract_columns(sub_batches)
    assert sub_batches_size.shape[1] % world_size == 0 , f"error: sub_batches_size is not correct, shape: {sub_batches_size.shape}"
    sub_batches_num = sub_batches_size.shape[1] // world_size
    # 构造在 lessjp 前的 model_trace
    model_trace = np.array([[(j-i+world_size) % world_size for i in range(world_size)]*sub_batches_num for j in range(world_size)]) # 因为模型左移，所以之前的 (i+j)%world_size 错误
    
    if jp_times > 0:
        for lesstimes in range(world_size - jp_times):
            # 每 group_size 个列里，去掉 1 列 （表示减少了跳的次数）
            group_size = world_size - lesstimes
            col_min = np.min(sub_batches_size, axis=0)
            reshaped_col_min = col_min.reshape(-1, group_size)
            min_indices = np.argmin(reshaped_col_min, axis=1).flatten()
            min_indices += np.arange(len(min_indices))*group_size # 加上其所在下标 *group_size
            # 找出每个列的最小值，从 model_trace 去掉最小值中最小的所在的列
            keep_cols = np.ones(model_trace.shape[1], dtype=bool) # 创建 bool 数组标记要保留的列
            keep_cols[min_indices] = False
            model_trace = model_trace[:, keep_cols]
            sub_batches_size = sub_batches_size[:, keep_cols] # 删除掉部分列
    else:
        # jp_times == 0 meas no jp
        model_trace = np.array([[j]*sub_batches_num for j in range(world_size)])
    return model_trace

def get_sub_batchs(epoch, fg_train_nid, world_size, ntrain_per_gpu, graph, rank, split_fn, jp_times, nid2subpid):
    np.random.seed(epoch)
    np.random.shuffle(fg_train_nid)
    useful_fg_train_nid = fg_train_nid[:world_size*ntrain_per_gpu]
    useful_fg_train_nid = useful_fg_train_nid.reshape(world_size, ntrain_per_gpu) # 每行表示一个gpu要训练的epoch train nid
    # logging.debug(f'rank: {rank} useful_fg_train_nid:{useful_fg_train_nid}')

    # 根据args.batch_size将每行切分为多个batch，然后转置，最终结果类似：array([[array([0, 1]), array([5, 6])], [array([2, 3]), array([7, 8])], [array([4]), array([9])]], dtype=object)；行数表示batch数量
    # useful_fg_train_nid = np.apply_along_axis(split_fn, 1, useful_fg_train_nid,).T
    # logging.debug(f'rank:{rank} useful_fg_train_nid.split.T:{useful_fg_train_nid}')

    # 使用列表推导式创建对象数组
    batch_list = []
    for i in range(useful_fg_train_nid.shape[0]):
        batches = split_fn(useful_fg_train_nid[i])
        batch_list.append(batches)
    # 创建对象数组并转置
    useful_fg_train_nid = np.array(batch_list, dtype=object).T
    # if rank == 0:
    #     print("=" * 50)
    #     print("输出结构分析:")
    #     print("=" * 50)
    #     print(f"总批次数量 (维度0): {useful_fg_train_nid.shape[0]}")
    #     print(f"GPU数量 (维度1): {useful_fg_train_nid.shape[1]}")
    #     # 查看前几个批次的结构
    #     for batch_idx in range(min(3, useful_fg_train_nid.shape[0])):
    #         print(f"\n批次 {batch_idx}:")
    #         for gpu_idx in range(useful_fg_train_nid.shape[1]):
    #             batch_data = useful_fg_train_nid[batch_idx, gpu_idx]
    #             print(f"  GPU {gpu_idx}: 形状={batch_data.shape}, 长度={len(batch_data)}, 前5个元素={batch_data[:5]}")
        
    ########## 确定该gpu在当前epoch中将要训练的所有sub-batch的nid，放入sub_batch_nid中，同时构建sub_batch_offsets，以备NeighborSamplerWithDiffBatchSz中使用 ###########
    # cache_partidx = cache_client.get_cache_partid()
    # assert cache_partidx == rank, 'rank设置需要与partidx相同，否则影响命中率'

    # 使用DGL的分区书替代原有的cache_client和nid2pid
    partition_book = graph.get_partition_book()
    current_part_id = rank  # 在DGL中，rank通常对应partition id
        
    sub_batch_nid = []
    sub_batch_offsets = [0]
    cur_offset = 0
    # 为确保每个 model 学习对应 mini-batch 的训练数据，需要根据交换列的顺序
    reversed_useful_fg_train_nid = reverse_columns(useful_fg_train_nid, rank)
    for row in reversed_useful_fg_train_nid:
        for batch in row:
            batch_tensor = th.tensor(batch, dtype=th.long)
            # part_ids = partition_book.nid2partid(batch_tensor)
            part_ids = nid2subpid[batch_tensor]
            cur_gpu_nid_mask = (part_ids == current_part_id)
            sub_batch = batch[cur_gpu_nid_mask]
            sub_batch_nid.extend(sub_batch)
            cur_offset += len(sub_batch)
            sub_batch_offsets.append(cur_offset)
            # cur_gpu_nid_mask = (nid2pid[batch]==cache_partidx)
            # sub_batch = batch[cur_gpu_nid_mask]
            # sub_batch_nid.extend(sub_batch)
            # cur_offset += len(sub_batch)
            # sub_batch_offsets.append(cur_offset)
            # if gpuid==0:
            #     logging.debug(f'put sub_batch: {batch[cur_gpu_nid_mask]}')
    ##### 计算 jp_times 时的模型迁移路径 #####
    model_trace = get_model_trace(jp_times, world_size, sub_batch_offsets)
    # logging.debug(f'model_trace: {model_trace}')

        
    ##### 根据 model_trace, 当前rank 对每个 iteration 中的 sub_batches 重新划分 #####
    # 将 model_trace 转置， 找出每行中值为当前 rank 所在的列的 indices 列表，表示将移动到本节点的模型 id 顺序
    model_trace = model_trace.T
    find_col_indices = np.argwhere(model_trace == rank)[:, 1]
    # 每 jp_times 为一组，表示一个 iteration 中每个 sub_batch 来源的 batch id (针对没有反转过的 useful_fg_train_nid)
    # 据此，重新构造 sub_batch_nid 和 sub_batch_offsets
    new_sub_batch_nids = []
    new_sub_batch_offsets = [0]
    new_cur_offsets = 0
    for rowid, row in enumerate(useful_fg_train_nid): # one interation for one row
        # 原本在当前节点要训练的各个 sub_batch 
        total_nodes = 0
        ori_sub_batch = []
        for batch in row:
            batch_tensor = th.tensor(batch, dtype=th.long)
            # part_ids = partition_book.nid2partid(batch_tensor)
            part_ids = nid2subpid[batch_tensor]
            cur_gpu_nid_mask = (part_ids == current_part_id)
            sub_batch = batch[cur_gpu_nid_mask]
            ori_sub_batch.append(sub_batch)
            total_nodes += len(sub_batch)
        ori_sub_batch = np.array(ori_sub_batch, dtype=object) # convert from list into numpy array
        # lessjp后被选中的 sub_batch 存储为 keep_sub_batches
        batchids = find_col_indices[rowid*jp_times:(rowid+1)*jp_times]
        keep_bool = np.zeros(world_size, dtype=bool)
        keep_bool[batchids] = True
        keep_sub_batches = ori_sub_batch[keep_bool]
        # logging.info(f'keep_sub_batches: {keep_sub_batches}')
        # 将未选中的 sub_batch 拆开分配到 keep_sub_batches
        unkeep_nids = []
        unkeep_batches = ori_sub_batch[~keep_bool]
        for unkeep_batch in unkeep_batches:
            unkeep_nids.extend(unkeep_batch)
        avg_len = total_nodes // len(batchids) + 1
        # logging.debug(f'avg_len: {avg_len}, unkeep_nids_len: {len(unkeep_nids)}')
        for sub_batch in keep_sub_batches:
            len_sub_batch = len(sub_batch)
            if len_sub_batch < avg_len and len(unkeep_nids) > 0:
                sub_batch = np.concatenate((sub_batch, unkeep_nids[ : avg_len - len_sub_batch]))
                unkeep_nids = unkeep_nids[avg_len - len_sub_batch:]
            # 添加到最终的结果中
            new_sub_batch_nids.extend(sub_batch)
            new_cur_offsets += len(sub_batch)
            new_sub_batch_offsets.append(new_cur_offsets)
        assert len(unkeep_nids)==0, 'unkeep_nids was not used up'
    # logging.debug(f'iteration {rowid} done, len nids {len(new_sub_batch_nids)}, unkeep_nids_len: {len(unkeep_nids)}')
    return new_sub_batch_nids, new_sub_batch_offsets, model_trace.T

class LeapGNNDataLoader:
    def __init__(self, graph, sub_batch_nids, sub_batch_offsets, sampler, device='cpu'):
        self.graph = graph
        # 确保节点ID是张量
        if isinstance(sub_batch_nids, list):
            self.sub_batch_nids = th.tensor(sub_batch_nids, dtype=th.long, device=device)
        else:
            self.sub_batch_nids = sub_batch_nids.to(device)
        self.sub_batch_offsets = sub_batch_offsets
        self.sampler = sampler
        self.num_batches = len(sub_batch_offsets) - 1
        self.device = device
        
    def __iter__(self):
        for i in range(self.num_batches):
            start = self.sub_batch_offsets[i]
            end = self.sub_batch_offsets[i + 1]
            batch_nodes = self.sub_batch_nids[start:end]
            
            # 确保batch_nodes是张量
            if isinstance(batch_nodes, list):
                batch_nodes = th.tensor(batch_nodes, dtype=th.long, device=self.device)
            
            # 使用采样器采样
            input_nodes, output_nodes, blocks = self.sampler.sample_blocks(
                self.graph, batch_nodes
            )
            
            yield input_nodes, output_nodes, blocks
            
    def __len__(self):
        return self.num_batches

def create_leapgnn_sampler(graph, sub_batch_nids, sub_batch_offsets, fanouts):
    """
    为LeapGNN创建自定义采样器
    """
    # 创建基础的邻居采样器
    sampler = dgl.dataloading.NeighborSampler([int(fanout) for fanout in args.fan_out.split(",")])
    
    # 将节点ID和偏移量转换为DGL可用的格式
    # sub_batch_offsets 应该是类似 [0, 10, 25, 40, ...] 的列表
    # sub_batch_nids 是所有子批次节点的扁平列表
    
    return LeapGNNDataLoader(
        graph, 
        sub_batch_nids, 
        sub_batch_offsets, 
        sampler
    )

def send_recv_model_trace_trace(model_trace, model, device, rank, jp_cnt, machine2model, world_size):
    # 根据当前机器上的 modelid 和 model_trace 确定应该哪个机器发送参数
    # 计算当前机器的模型下一步的目的机器id，如果等于当前机器id（即rank），则不用迁移
    cur_model_id = machine2model[rank]
    dst_machine_id = model_trace[cur_model_id][jp_cnt]
    # logging.debug(f"Start jump model the {jp_cnt}th times, cur_model_id={cur_model_id}, next_dst_machine_id={dst_machine_id}")

    # 确定发送方rank：找到哪个rank持有我们需要接收的模型
    src_machine_id = None
    for model_id in range(world_size):
        if model_trace[model_id][jp_cnt] == rank:
            src_machine_id = model_id
            break
    # 找到当前持有src_machine_id模型的rank
    src_rank = None
    for r in range(world_size):
        if machine2model[r] == src_machine_id:
            src_rank = r
            break

    # print(f"Rank {rank}: cur_model_id={cur_model_id}, dst_machine_id={dst_machine_id}, src_machine_id={src_machine_id}, src_rank={src_rank}")

    # 确定哪些机器发送数据，例如 0->2 1->3 2->0 3->1 则原始的奇偶交错发送的方法就会导致死锁
    chains_lst = []
    checked = [False for _ in range(world_size)]
    send_ranks = [] # rank to send first
    for tmprank in range(world_size):
        tmp_dst = model_trace[machine2model[tmprank]][jp_cnt]
        chains_lst.append((tmprank, tmp_dst))
    chains_lst.sort()
    for s,d in chains_lst:
        if checked[s] == False:
            send_ranks.append(s)
            checked[s] = True
            checked[d] = True # 源节点和目的结点不能同时 send，会死锁
        else:
            if checked[d] == False:
                send_ranks.append(d)
                checked[d] = True
    # logging.debug(f'chains_lst: {chains_lst} send_ranks: {send_ranks}')

    if dst_machine_id != rank:
        send_first = (rank in send_ranks)
        for val in model.parameters():
            new_val = th.zeros_like(val, device = device)
            if send_first:
                th.distributed.send(val, dst = dst_machine_id)
                th.distributed.recv(new_val, src = src_rank)
            else:
                th.distributed.recv(new_val, src = src_rank)
                th.distributed.send(val, dst = dst_machine_id)
            with th.no_grad():
                val[:] = new_val
    # logging.debug(f"End jump model")
    # update machine2model
    for rowid in range(world_size):
        machineid = model_trace[rowid][jp_cnt]
        machine2model[machineid] = rowid
    # logging.debug(f'cur machine2modelid: {machine2model}')

def create_sub_partitions_array(original_partitions, gpus_per_machine, world_size, num_nodes):

    num_machines = world_size // gpus_per_machine
    nid2subpid = np.full(num_nodes, -1, dtype=np.int64)
    
    for original_pid in range(num_machines):
        partition_nodes = original_partitions.partid2nids(original_pid).numpy()
        
        num_nodes = len(partition_nodes)
        nodes_per_subpartition = num_nodes // gpus_per_machine
        
        for i, node_id in enumerate(partition_nodes):
            sub_pid = i // nodes_per_subpartition
            if sub_pid >= gpus_per_machine:
                sub_pid = gpus_per_machine - 1
            global_subpid = original_pid * gpus_per_machine + sub_pid
            nid2subpid[node_id] = global_subpid
    
    return nid2subpid
#----------leapgnn-end-----------

class DistGAT(nn.Module):

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 n_heads,
                 activation=F.relu,
                 feat_dropout=0.6,
                 attn_dropout=0.6):
        assert len(n_heads) == n_layers
        assert n_heads[-1] == 1

        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_heads = n_heads

        self.layers = nn.ModuleList()
        for i in range(0, n_layers):
            in_dim = in_feats if i == 0 else n_hidden * n_heads[i - 1]
            out_dim = n_classes if i == n_layers - 1 else n_hidden
            layer_activation = None if i == n_layers - 1 else activation
            self.layers.append(
                dgl.nn.pytorch.GATConv(in_dim,
                              out_dim,
                              n_heads[i],
                              feat_drop=feat_dropout,
                              attn_drop=attn_dropout,
                              activation=layer_activation,
                              allow_zero_in_degree=True))

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i == self.n_layers - 1:
                h = h.mean(1)
            else:
                h = h.flatten(1)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GAT model on full neighbors (i.e. without
        neighbor sampling).

        g : the entire graph.
        x : the input of entire node set.

        Distributed layer-wise inference.
        """
        nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )

        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                y = dgl.distributed.DistTensor(
                    (g.num_nodes(), self.n_classes * self.n_heads[i]),
                    th.float32,
                    "h_last",
                    persistent=True,
                )
            else:
                y = dgl.distributed.DistTensor(
                    (g.num_nodes(), self.n_hidden * self.n_heads[i]),
                    th.float32,
                    "h",
                    persistent=True,
                )
            print(f"|V|={g.num_nodes()}, eval batch size: {batch_size}")

            sampler = dgl.dataloading.NeighborSampler([-1])
            dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i == self.n_layers - 1:
                    h = h.mean(1)
                else:
                    h = h.flatten(1)

                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return y


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def initializer(shape, dtype):
    arr = th.zeros(shape, dtype=dtype)
    arr.uniform_(-1, 1)
    return arr


class DistEmb(nn.Module):
    def __init__(
            self, num_nodes, emb_size, dgl_sparse_emb=False, dev_id="cpu"
    ):
        super().__init__()
        self.dev_id = dev_id
        self.emb_size = emb_size
        self.dgl_sparse_emb = dgl_sparse_emb
        if dgl_sparse_emb:
            self.sparse_emb = DistEmbedding(
                num_nodes, emb_size, name="sage", init_func=initializer
            )
        else:
            self.sparse_emb = th.nn.Embedding(num_nodes, emb_size, sparse=True)
            nn.init.uniform_(self.sparse_emb.weight, -1.0, 1.0)

    def forward(self, idx):
        # embeddings are stored in cpu
        idx = idx.cpu()
        if self.dgl_sparse_emb:
            return self.sparse_emb(idx, device=self.dev_id)
        else:
            return self.sparse_emb(idx).to(self.dev_id)


def load_embs(standalone, emb_layer, g):
    nodes = dgl.distributed.node_split(
        np.arange(g.num_nodes()), g.get_partition_book(), force_even=True
    )
    x = dgl.distributed.DistTensor(
        (
            g.num_nodes(),
            emb_layer.module.emb_size
            if isinstance(emb_layer, th.nn.parallel.DistributedDataParallel)
            else emb_layer.emb_size,
        ),
        th.float32,
        "eval_embs",
        persistent=True,
    )
    num_nodes = nodes.shape[0]
    for i in range((num_nodes + 1023) // 1024):
        idx = nodes[
            i * 1024: (i + 1) * 1024
            if (i + 1) * 1024 < num_nodes
            else num_nodes
        ]
        embeds = emb_layer(idx).cpu()
        x[idx] = embeds

    if not standalone:
        g.barrier()

    return x


def evaluate(
    standalone,
    model,
    emb_layer,
    g,
    labels,
    val_nid,
    test_nid,
    batch_size,
    device,
):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    if not standalone:
        model = model.module
    model.eval()
    emb_layer.eval()
    with th.no_grad():
        inputs = load_embs(standalone, emb_layer, g)
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    emb_layer.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(
        pred[test_nid], labels[test_nid]
    )


def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, n_classes, g = data

    #----------leapgnn-start-----------

    #################### 读取全图中的训练点id、计算每个gpu需要训练的nid数量
    fg_train_nid = th.where(g.ndata["train_mask"][th.arange(g.num_nodes())])[0].numpy()
    ntrain_per_gpu = int(fg_train_nid.shape[0] / th.distributed.get_world_size())
    print("rank {}, ntrain_per_gpu {}".format(
        th.distributed.get_rank(),
        ntrain_per_gpu
    ))


    def split_fn(a):
        if len(a) <= args.batch_size:
            return np.split(a, np.array([len(a)])) # 最后会产生一个空的array, necessary
        else:
            return np.split(a, np.arange(args.batch_size, len(a), args.batch_size)) 
            # 例如a=[0,1, ..., 9]，bs=3，那么切割的结果是[0,1,2], [3,4,5], [6,7,8], [9]

    #----------leapgnn-end-----------

    # sampler = dgl.dataloading.NeighborSampler(
    #     [int(fanout) for fanout in args.fan_out.split(",")]
    # )
    # dataloader = dgl.dataloading.DistNodeDataLoader(
    #     g,
    #     train_nid,
    #     sampler,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     drop_last=False,
    # )

    # Define model and optimizer
    emb_layer = DistEmb(
        g.num_nodes(),
        args.num_emb,
        dgl_sparse_emb=args.dgl_sparse,
        dev_id=device,
    )
    if args.model == 'sage':
        model = DistSAGE(
            args.num_emb,
            args.num_hidden,
            n_classes,
            args.num_layers,
            F.relu,
            args.dropout,
        )
    elif args.model == 'gat':
        gat_heads = [int(head) for head in args.heads.split(",")]
        model = DistGAT(
            args.num_emb,
            args.num_hidden,
            n_classes,
            args.num_layers,
            gat_heads,
            F.relu,
            args.dropout
        )
    model = model.to(device)
    if not args.standalone:
        if args.num_gpus == -1:
            model = th.nn.parallel.DistributedDataParallel(model)
        else:
            dev_id = th.distributed.get_rank() % args.num_gpus
            model = th.nn.parallel.DistributedDataParallel(
                model, device_ids=[dev_id], output_device=dev_id
            )
            if not args.dgl_sparse:
                emb_layer = th.nn.parallel.DistributedDataParallel(emb_layer)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.dgl_sparse:
        # emb_optimizer = dgl.distributed.optim.SparseAdam(
        #     [emb_layer.sparse_emb], lr=args.sparse_lr
        # )
        emb_optimizer = SparseAdam(
            [emb_layer.sparse_emb], lr=args.sparse_lr
        )
        print("optimize DGL sparse embedding:", emb_layer.sparse_emb)
    elif args.standalone:
        emb_optimizer = th.optim.SparseAdam(
            list(emb_layer.sparse_emb.parameters()), lr=args.sparse_lr
        )
        print("optimize Pytorch sparse embedding:", emb_layer.sparse_emb)
    else:
        emb_optimizer = th.optim.SparseAdam(
            list(emb_layer.module.sparse_emb.parameters()), lr=args.sparse_lr
        )
        print(
            "optimize Pytorch sparse embedding:",
            emb_layer.module.sparse_emb
        )

    # Training loop
    iter_tput = []
    epoch = 0
    for epoch in range(args.num_epochs):
        tic = time.time()

        sample_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        start = time.time()
    #----------leapgnn-start-----------

        world_size = th.distributed.get_world_size()
        rank = th.distributed.get_rank()
        th.cuda.set_device(rank % args.num_gpus)
        jp_times = world_size # 跳跃次数
        moniter_epoch_time = [] # moniter the avg. epoch time
        all_epoch_time = [] # moniter each epoch time for exp details
        adjust_jp_times = True # 是否继续缩小 jp_times
        miss_rate_lst = [None for _ in range(world_size)] # each rank's miss rate, (rank0, 1, 2, ...)
        machine2model = [i for i in range(world_size)] # 记录随着模型的迁移，每个机器(按顺序)上对应的 model_id, 初始状态下机器id=model_id
        last_avg_epoch_time = float('inf')
        nid2subpid = create_sub_partitions_array(
            g.get_partition_book(), 
            args.num_gpus, 
            world_size, 
            g.num_nodes()
        )

        print("get nid2subpid")

        if len(moniter_epoch_time) >= args.moniter_epochs and adjust_jp_times==True:
            avg_epoch_time = sum(moniter_epoch_time) / len(moniter_epoch_time)
            if avg_epoch_time < last_avg_epoch_time:
                if th.distributed.get_rank() == 0:
                    print(f"avg_epoch_time of {moniter_epoch_time} is {avg_epoch_time} which is smaller than last avg epoch time {last_avg_epoch_time}")
                jp_times -= 1
                if th.distributed.get_rank() == 0:
                    print(f"new jp_times is {jp_times}")
                if jp_times == 0:
                    jp_times += 1 # 并不算完全退化，保留第一次的跳跃权利
                    adjust_jp_times = False # 保持一跳的方式
                moniter_epoch_time = []
                last_avg_epoch_time = avg_epoch_time # update last_avg_epoch_time
            else:
                if th.distributed.get_rank() == 0:
                    print(f"avg_epoch_time of {moniter_epoch_time} is {avg_epoch_time} which is larger than last avg epoch time {last_avg_epoch_time}")
                adjust_jp_times = False
                jp_times += 1 # 恢复性能更好的jp_times


        if jp_times > 0:
            ########## 确定当前epoch每个gpu要训练的batch nid ###########
            sub_batch_nids, sub_batch_offsets, model_trace = get_sub_batchs(
                epoch, 
                fg_train_nid, 
                world_size, 
                ntrain_per_gpu, 
                g,
                rank,
                split_fn, 
                jp_times, 
                nid2subpid
            )
            if th.distributed.get_rank() == 0:
                print(f'get_sub_batchs done')
                
            ########## 根据分配到的sub_batch_nid和sub_batch_offsets，构造采样器 ###########
            # with sem:
            #     sampler = dgl.contrib.sampling.NeighborSamplerWithDiffBatchSz(fg, sub_batch_offsets, expand_factor=int(sampling[0]), num_hops=len(sampling)+1, neighbor_type='in', shuffle=False, num_workers=args.num_worker, seed_nodes=sub_batch_nid, prefetch=True, add_self_loop=True)
            #     if th.distributed.get_rank() == 0:
            #         print(f'init sampler done')

            dataloader = create_leapgnn_sampler(g, sub_batch_nids, sub_batch_offsets, args.fan_out)
            print("use leapgnn train")

        else:
            # 退化为 default 模式
            # np.random.seed(epoch)
            # np.random.shuffle(fg_train_nid)
            # train_lnid = fg_train_nid[args.rank * ntrain_per_gpu: (args.rank+1)*ntrain_per_gpu]
            # sampler = dgl.contrib.sampling.NeighborSampler(fg, args.batch_size, expand_factor=int(sampling[0]), num_hops=len(sampling)+1, neighbor_type='in', shuffle=True, num_workers=args.num_worker, seed_nodes=train_lnid, prefetch=True, add_self_loop=True)
            sampler = dgl.dataloading.NeighborSampler(
                [int(fanout) for fanout in args.fan_out.split(",")]
            )
            dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                train_nid,
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
            )
            print("use dgl train")

    #----------leapgnn-end-----------
        with model.join():
            # Loop over the dataloader to sample the computation dependency
            # graph as a list of blocks.
            step_time = []
    #----------leapgnn-start-----------
            jp_cnt = 0
            step = 0
            sampler_iterator = iter(dataloader)
            # for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            for sub_nf_id in range(len(sub_batch_offsets)-1):
                if jp_times > 0:
                    ##### 在一个 sub_batch 训练开始前，迁移模型 #####
                    send_recv_model_trace_trace(model_trace, model, device, rank, jp_cnt, machine2model, world_size)
                    jp_cnt += 1
                    ########## 获取sub_nfs，跨结点获取sub_nfs的feature数据 ###########
                    if sub_nf_id % jp_times == 0:
                        # 一次获取world_size个nf，进行预取
                        sub_nfs_lst = [] # 存放提前预取的包含features的nf
                        for j in range(jp_times):
                            try:
                                sub_nfs_lst.append(next(sampler_iterator)) # 获取子图topo
                            except StopIteration:
                                continue # 可能不足batch size个
                        # with torch.autograd.profiler.record_function('fetch feat'):
                        #     fetch_func(sub_nfs_lst) # 获取feats存入sub_nfs_lst列中中的对象属性    
                    # 选择其中一个sub_nf参与后续计算
                    input_nodes, seeds, blocks = sub_nfs_lst[sub_nf_id%jp_times]
                else:
                    input_nodes, seeds, blocks = next(sampler_iterator)
    #----------leapgnn-ends-----------
                tic_step = time.time()
                sample_time += tic_step - start
                num_seeds += len(blocks[-1].dstdata[dgl.NID])
                num_inputs += len(blocks[0].srcdata[dgl.NID])
                blocks = [block.to(device) for block in blocks]
                batch_labels = g.ndata["labels"][seeds].long().to(device)
                # Compute loss and prediction
                start = time.time()
                batch_inputs = emb_layer(input_nodes)
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                forward_end = time.time()
                emb_optimizer.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                compute_end = time.time()
                forward_time += forward_end - start
                backward_time += compute_end - forward_end

                emb_optimizer.step(args.num_gpus)
                optimizer.step()
                update_time += time.time() - compute_end

                step_t = time.time() - tic_step
                step_time.append(step_t)
                iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
                if step % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    gpu_mem_alloc = (
                        th.cuda.max_memory_allocated() / 1000000
                        if th.cuda.is_available()
                        else 0
                    )
                    print(
                        "Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | "
                        "Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU "
                        "{:.1f} MB | time {:.3f} s".format(
                            g.rank(),
                            epoch,
                            step,
                            loss.item(),
                            acc.item(),
                            np.mean(iter_tput[3:]),
                            gpu_mem_alloc,
                            np.sum(step_time[-args.log_every:]),
                        )
                    )
                step += 1
                start = time.time()

        toc = time.time()
        print(
            "Part {}, Epoch Time(s): {:.4f}, sample+data_copy: {:.4f}, forward"
            ": {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, #inputs"
            ": {}".format(
                g.rank(),
                toc - tic,
                sample_time,
                forward_time,
                backward_time,
                update_time,
                num_seeds,
                num_inputs,
            )
        )
        moniter_epoch_time.append(toc - tic)
        epoch += 1

        if epoch % args.eval_every == 0 and epoch != 0:
            start = time.time()
            val_acc, test_acc = evaluate(
                args.standalone,
                model,
                emb_layer,
                g,
                g.ndata["labels"],
                val_nid,
                test_nid,
                args.batch_size_eval,
                device,
            )
            print(
                "Part {}, Val Acc {:.4f}, Test Acc {:.4f}, time: {:.4f}".format
                (
                    g.rank(), val_acc, test_acc, time.time() - start
                )
            )


def main(args):
    dgl.distributed.initialize(args.ip_config)
    if not args.standalone:
        os.environ['NCCL_SOCKET_IFNAME'] = 'ens5f0'
        os.environ['NCCL_IB_DISABLE'] = '1'
        th.distributed.init_process_group(backend="nccl")
    g = dgl.distributed.DistGraph(
            args.graph_name,
            part_config=args.part_config
        )
    print("rank:", g.rank())

    pb = g.get_partition_book()
    train_nid = dgl.distributed.node_split(
        g.ndata["train_mask"], pb, force_even=True
    )
    val_nid = dgl.distributed.node_split(
        g.ndata["val_mask"], pb, force_even=True
    )
    test_nid = dgl.distributed.node_split(
        g.ndata["test_mask"], pb, force_even=True
    )
    local_nid = pb.partid2nids(pb.partid).detach().numpy()

    print(
        "part {}, train: {} (local: {}), val: {} (local: {}), test: {} "
        "(local: {})".format(
            g.rank(),
            len(train_nid),
            len(np.intersect1d(train_nid.numpy(), local_nid)),
            len(val_nid),
            len(np.intersect1d(val_nid.numpy(), local_nid)),
            len(test_nid),
            len(np.intersect1d(test_nid.numpy(), local_nid)),
        )
    )
    if args.num_gpus == -1:
        device = th.device("cpu")
    else:
        dev_id = th.distributed.get_rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
        print("cuda:"+str(dev_id))
    labels = g.ndata["labels"][np.arange(g.num_nodes())]
    n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    print("#labels:", n_classes)

    # Pack data
    data = train_nid, val_nid, test_nid, n_classes, g
    run(args, device, data)
    print("parent ends")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--id", type=int, help="the partition id")
    parser.add_argument(
        "--ip_config", type=str, help="The file for IP configuration"
    )
    parser.add_argument(
        "--part_config", type=str, help="The path to the partition config file"
    )
    parser.add_argument("--n_classes", type=int, help="the number of classes")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,25")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--local_rank", type=int, help="get rank of the process"
    )
    parser.add_argument(
        "--standalone", action="store_true", help="run in the standalone mode"
    )
    parser.add_argument(
        "--dgl_sparse",
        action="store_true",
        help="Whether to use DGL sparse embedding",
    )
    parser.add_argument(
        "--sparse_lr", type=float, default=1e-2, help="sparse lr rate"
    )
    parser.add_argument(
        '--moniter_epochs', default=1, type=int, 
        help='number of epochs to moniter each epoch training time'
    )
    parser.add_argument(
        '--model', default='sage', type=str
    )
    parser.add_argument("--heads", type=str, default="8,8,1")
    parser.add_argument("--num_emb", type=int, default=256)
    args = parser.parse_args()

    print(args)
    main(args)
