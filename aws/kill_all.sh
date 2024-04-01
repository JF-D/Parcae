#!/bin/bash
sudo pkill -f etcd

nnode=${1:-0}
ip_file=${2:-"hostname"}
machines=$(cat $ip_file)


for node in $machines
do
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@${node} "sudo pkill -9 -f pretrain_gpt2.py; sudo pkill -9 -f project_pactum.run" &
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@${node} "nvidia-smi | grep python | awk '{ print \$5 }' | xargs -n1 kill -9" &
    nnode=$((nnode-1))
    if [ $nnode -eq 0 ]; then
        break
    fi
done

sleep 1
