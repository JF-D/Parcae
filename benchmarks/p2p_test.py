import os
import re
import time
import argparse
import numpy as np
import torch
import torch.distributed as dist


parser = argparse.ArgumentParser()
parser.add_argument('-type', type=str, default='all_reduce')
parser.add_argument('-master_addr', type=str, default='localhost')
parser.add_argument('-port', type=str, default='12355')
parser.add_argument('--backend', type=str, default='nccl')
parser.add_argument('--slurm', action='store_true')
parser.add_argument('-launch', type=str, default='slurm')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()


def setup(local_rank, rank, world_size):
    # initialize the process group
    torch.cuda.set_device(local_rank)
    if args.launch == 'deepspeed':
        import deepspeed
        deepspeed.init_distributed()
    else:
        dist.init_process_group(args.backend)
    print(f'rank {rank}({local_rank}) init with {world_size} nccl process')


def cleanup():
    dist.destroy_process_group()


def sync_all():
    torch.cuda.synchronize()
    dist.barrier()


def p2p_latency(nbytes):
    buf = torch.randn(nbytes // 4)
    if args.backend == 'nccl':
        buf = buf.cuda()

    sync_all()
    # warmup
    for _ in range(5):
        if args.rank == 0:
            dist.send(buf, 1)
        else:
            dist.recv(buf, 0)
    sync_all()

    sync_all()
    begin = time.perf_counter()
    for _ in range(25):
        if args.rank == 0:
            dist.send(buf, 1)
        else:
            dist.recv(buf, 0)
    sync_all()
    end = time.perf_counter()
    avg_speed = (end - begin) * 1e6 / 25

    iter_speeds = []
    for _ in range(25):
        sync_all()
        begin = time.perf_counter()
        if args.rank == 0:
            dist.send(buf, 1)
        else:
            dist.recv(buf, 0)
        sync_all()
        end = time.perf_counter()
        iter_speeds.append((end - begin) * 1e6)

    if args.rank == 0:
        print(
            f'{nbytes:15d}({nbytes / 1024 / 1024:8.2f}MB): {avg_speed:8.3f}us {min(iter_speeds):8.3f}us'
        )


def p2p_perf():
    tensor_sizes = [2**i for i in range(2, 31)]
    for payload in tensor_sizes:
        p2p_latency(payload)


if __name__ == '__main__':
    if args.launch == 'slurm':
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        node_list = str(os.environ['SLURM_NODELIST'])
        node_parts = re.findall('[0-9]+', node_list)

        os.environ[
            'MASTER_ADDR'] = f'{node_parts[1]}.{node_parts[2]}.{node_parts[3]}.{node_parts[4]}'
        os.environ['MASTER_PORT'] = str(args.port)
    elif args.launch == 'mpirun':
        local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
        rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
        world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.port
        print(rank, world_size, local_rank, os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
    elif args.launch == 'deepspeed':
        local_rank = args.local_rank
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank
    setup(local_rank, rank, world_size)

    p2p_perf()

    cleanup()

'''
CMD: mpirun -np 2 --host ip-172-31-28-108,ip-172-31-8-205 -mca btl_tcp_if_include ens3 /opt/conda/envs/pytorch/bin/python p2p_test.py -master_addr ip-172-31-28-108 -port 12365 -launch mpirun
'''
