python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 256 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'sage' \
--num_emb 256" >> sage-products.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 32 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'gat' \
--num_emb 256" >> gat-products.log

## 下面的part_config和graph_name要换成papers100M
python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 256 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'sage' \
--num_emb 128" >> sage-papers.log

## 下面的part_config和graph_name要换成papers100M
python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 32 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'gat' \
--num_emb 128" >> gat-papers.log

## 下面的part_config和graph_name要换成mag
python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 256 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'sage' \
--num_emb 128" >> sage-mag.log

## 下面的part_config和graph_name要换成mag
python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 32 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'gat' \
--num_emb 128" >> gat-mag.log

## 测试hidden dim sage products
python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 256 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'sage' \
--num_emb 256" >> sage-products-256.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 192 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'sage' \
--num_emb 256" >> sage-products-192.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 128 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'sage' \
--num_emb 256" >> sage-products-128.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 96 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'sage' \
--num_emb 256" >> sage-products-96.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 64 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'sage' \
--num_emb 256" >> sage-products-64.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 32 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'sage' \
--num_emb 256" >> sage-products-32.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 16 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'sage' \
--num_emb 256" >> sage-products-16.log

## 测试hidden dim gat products
python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 32 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'gat' \
--num_emb 256" >> gat-products-256.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 24 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'gat' \
--num_emb 256" >> gat-products-192.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 16 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'gat' \
--num_emb 256" >> gat-products-128.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 12 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'gat' \
--num_emb 256" >> gat-products-96.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 8 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'gat' \
--num_emb 256" >> gat-products-64.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 4 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'gat' \
--num_emb 256" >> gat-products-32.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 2 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'gat' \
--num_emb 256" >> gat-products-16.log

## 测试hidden dim sage papers
python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 256 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'sage' \
--num_emb 256" >> sage-papers-256.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 192 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'sage' \
--num_emb 256" >> sage-papers-192.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 128 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'sage' \
--num_emb 256" >> sage-papers-128.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 96 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'sage' \
--num_emb 256" >> sage-papers-96.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 64 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'sage' \
--num_emb 256" >> sage-papers-64.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 32 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'sage' \
--num_emb 256" >> sage-papers-32.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 16 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'sage' \
--num_emb 256" >> sage-papers-16.log

## 测试hidden dim gat papers
python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 32 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'gat' \
--num_emb 256" >> gat-papers-256.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 24 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'gat' \
--num_emb 256" >> gat-papers-192.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 16 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'gat' \
--num_emb 256" >> gat-papers-128.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 12 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'gat' \
--num_emb 256" >> gat-papers-96.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 8 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'gat' \
--num_emb 256" >> gat-papers-64.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 4 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'gat' \
--num_emb 256" >> gat-papers-32.log

python3 ~/dgl113/dgl/tools/launch.py \
--workspace ~/dgl113/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /nfs/cryang/dgldataset/dist_ogbn_products/ogb-product.json \
--ip_config ip_config.txt \
"/home/cryang/anaconda3/envs/dglbuild/bin/python \
train_dist_transductive_leapgnn_multigpu.py \
--graph_name ogb-product \
--ip_config ip_config.txt \
--num_epochs 20 \
--batch_size 8000 \
--num_gpus 8 \
--num_hidden 2 \
--fan_out '5,10,15' \
--num_layers 3 \
--dgl_sparse \
--model 'gat' \
--num_emb 256" >> gat-papers-16.log

