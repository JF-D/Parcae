#!/usr/bin/bash
export PYTHONPATH=/home/ubuntu/Parcae:/opt/conda/envs/pytorch/bin:${PYTHONPATH}
# export PROJECT_PACTUM_LOGGING_WARNING='etcd.client,etcd.lock,torch.distributed.distributed_c10d'

pkill -f etcd
rm -rf /tmp/torchelastic_*
rm -rf /tmp/tmp_etcd/*

nnode=$1
tracefile=$2
logtag=${3:-"test"}
HOSTFILE="hostname"
DRY_RUN= #"--dry-run"

# APPROACH="liveput-truth"
APPROACH="liveput-predict"

# >>> first sync code to all nodes
python sync_code.py --n ${nnode} --hostfile ${HOSTFILE} ${DRY_RUN}


approach_tag=$( echo "${APPROACH}" | sed -r 's/-/_/g' )
logfile="train_${logtag}_${approach_tag}.log"
replayer_logfile="log/replayer_${logtag}_${approach_tag}.log"


cmd="python replay_trace.py --trace ${tracefile} \
    --n ${nnode} --hostfile ${HOSTFILE} ${DRY_RUN} \
    --approach ${APPROACH} \
    --rdzv_backend=etcd-v2 \
    --rdzv-ip=172.31.28.108 \
    --rdzv-port=2134 \
    --rdzv_conf='last_call_timeout=5,timeout=6000' \
    --nnodes=1:32 \
    --replayer-log ${replayer_logfile} \
    2>&1 | tee log/${logfile} \
"
echo ${cmd}
eval ${cmd}
