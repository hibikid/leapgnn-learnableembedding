#!/bin/bash

FILE="ip_config.txt"
FILE2="train_dist_transductive.py"
FILE3="train_dist.py"
FILE4="train_dist_transductive_leapgnn_multigpu.py"
FILE5="train_dist_transductive_leapgnn.py"
FILE6="local_utils.py"
FILE7="local_emboptimizer.py"

TARGET_PATH=$(pwd)

while IFS= read -r IP_ADDRESS; do
    echo "Sending $FILE to $IP_ADDRESS..."
    scp -r $FILE $FILE2 $FILE3 $FILE4 $FILE5 $FILE6 $FILE7 $IP_ADDRESS:$TARGET_PATH
    echo "File sent to $IP_ADDRESS"
done < $FILE