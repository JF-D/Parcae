#! /bin/bash
SCRIPT_PATH=$(dirname "$0")
cd $SCRIPT_PATH/..

export PYTHONPATH=/home/ubuntu/Parcae:/opt/conda/envs/pytorch/bin:${PYTHONPATH}
export PROJECT_PACTUM_LOGGING_WARNING='etcd.client,etcd.lock,torch.distributed.distributed_c10d'
export NCCL_SOCKET_IFNAME=ens3
export CHECKPOINT_PATH=checkpoints/gpt

rm -rf checkpoints/gpt/*
mkdir -p log

RDZV_IP=${RDZV_IP:-"127.0.0.1"}
RDZV_PORT=${RDZV_PORT:-2134}
RDZV_ID=${RDZV_ID:-"c63361db-da0b-40ca-90a6-a59be99e2a0b"}
RDZV_CONF=${RDZV_CONF:-"last_call_timeout=5,timeout=1000"}

GPU_PER_NODE=1


DATA_PATH=/home/ubuntu/datas/wikitext-2
VOCAB_PATH=data/gpt2/vocab.json
MERGE_PATH=data/gpt2/merges.txt
# CHECKPOINT_PATH=checkpoints/gpt


model="gpt-1.5b"
# model="gpt-6.7b"

#Actication Checkpointing and Contigious Memory
chkp_layers=0
DATAIMPL="synthetic"

if [ "${model}" = "gpt-1.5b" ]; then
    NLAYERS=48
    NHIDDEN=1600
    NHEADS=25
    BATCHSIZE=1
    SEQ_LENGTH=1024
    DATAIMPL="synthetic"
    export CPU_TRANSFER=OFF
elif [ "${model}" = "gpt-2" ]; then
    NLAYERS=24
    NHIDDEN=1024
    NHEADS=16
    BATCHSIZE=1
    SEQ_LENGTH=1024
elif [ "${model}" = "gpt-6.7b" ]; then
    export CPU_TRANSFER=ON
    NLAYERS=32
    NHIDDEN=4096
    NHEADS=32
    BATCHSIZE=1
    SEQ_LENGTH=1024
    chkp_layers=1
elif [ "${model}" = "gpt-8.3b" ]; then
    NLAYERS=72
    NHIDDEN=3072
    NHEADS=24
    BATCHSIZE=1
    SEQ_LENGTH=1024
    chkp_layers=1
elif [ "${model}" = "gpt-7.5b" ]; then
    export CPU_TRANSFER=ON
    NLAYERS=36
    NHIDDEN=4096
    NHEADS=32
    BATCHSIZE=1
    SEQ_LENGTH=1024
    chkp_layers=1
    DATAIMPL="wikitext"
elif [ "${model}" = "gpt-18.4b" ]; then
    NLAYERS=40
    NHIDDEN=6144
    NHEADS=48
    BATCHSIZE=1
    SEQ_LENGTH=2048
elif [ "${model}" = "gpt-175b" ]; then
    NLAYERS=96
    NHIDDEN=12288
    NHEADS=96
    BATCHSIZE=1
    SEQ_LENGTH=2048
else
    NLAYERS=1
    NHIDDEN=1600
    NHEADS=25
    BATCHSIZE=2
    SEQ_LENGTH=1024
fi

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
config_json="$script_dir/spotdl_${model}.json"

GAS=64 # gradient accumulation steps


gpt_options=" \
        --num-layers $NLAYERS \
        --hidden-size $NHIDDEN \
        --num-attention-heads $NHEADS \
        --seq-length ${SEQ_LENGTH} \
        --max-position-embeddings ${SEQ_LENGTH} \
        --batch-size $BATCHSIZE \
        --gas $GAS \
        --train-iters 5000 \
        --lr-decay-iters 320000 \
        --use-cpu-initialization \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_PATH \
        --merge-file $MERGE_PATH \
        --data-impl ${DATAIMPL} \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 1.5e-4 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup 0.01 \
        --log-interval 50 \
        --save-interval 500 \
        --eval-interval 10000 \
        --eval-iters 10 \
        --fp16 \
"

deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
            "

if [ ${chkp_layers} -ge 1 ]; then
    gpt_options="${gpt_options} --checkpoint-activations "
    chkp_opt=" \
        --checkpoint-activations \
        --checkpoint-num-layers ${chkp_layers}"
else
    chkp_opt=""
fi


full_options="${gpt_options} ${deepspeed_options} ${chkp_opt}"

# run on aws with mpirun
# >>> mpirun -np 4 --host ip-172-31-28-108,ip-172-31-8-205,ip-172-31-14-37,ip-172-31-20-46 -mca btl_tcp_if_include ens3 sh examples/spotdl.sh

# run on slurm
# run_cmd="srun -p caif_debug --preempt -N 4 --tasks-per-node 1 --gres=gpu:1 --mem-per-gpu=20GB python -m project_pactum.run \

# run_cmd="CUDA_VISIBLE_DEVICES=0 python -m project_pactum.run \
run_cmd="/opt/conda/envs/pytorch/bin/python -m project_pactum.run \
            --rdzv_backend=etcd-v2 \
            --rdzv_endpoint=$RDZV_IP:$RDZV_PORT \
            --rdzv_id=$RDZV_ID \
            --rdzv_conf=$RDZV_CONF \
            --nnodes=1:32 \
            --nproc_per_node=${GPU_PER_NODE} \
            --project-pactum \
            --max-pipe-parallel-size=24 \
            --default-num-stages=1 \
            pretrain_gpt2.py \
            $@ ${full_options} \
            2>&1 | tee log/log.log"
echo ${run_cmd}
eval ${run_cmd}

set +x
