# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron initialization."""

import random
import os
import functools

import numpy as np
import torch

from megatron import get_adlr_autoresume
from megatron import get_args
from megatron import get_tensorboard_writer
from megatron import mpu
from megatron.global_vars import set_global_variables
from megatron.mpu import set_model_parallel_rank, set_model_parallel_world_size

import deepspeed


def initialize_megatron(extra_args_provider=None, args_defaults={},
                        ignore_unknown_args=False, allow_no_cuda=False):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)

"""
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), 'Megatron requires CUDA.'

    # Parse args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(extra_args_provider=extra_args_provider,
                         args_defaults=args_defaults,
                         ignore_unknown_args=ignore_unknown_args)

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed()

        # Random seeds for reproducibility.
        if args.rank == 0:
            print('> setting random seeds to {} ...'.format(args.seed))
        _set_random_seed(args.seed)

    args = get_args()
    if  args.lazy_mpu_init:
        args.use_cpu_initialization=True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        set_model_parallel_world_size(args.model_parallel_size)
        # and return function for external DDP manager to call when it has DDP initialized
        set_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Initialize memory buffers.
        _initialize_mem_buffs()

        # Autoresume.
        _init_autoresume()

        # Write arguments to tensorboard.
        _write_args_to_tensorboard()
        # No continuation function
        return None


def setup_deepspeed_random_and_activation_checkpointing(args):
    '''Optional DeepSpeed Activation Checkpointing features.
    Gives access to partition activations, contiguous memory optimizations
    and cpu checkpointing.

    Activation checkpoint requires keep track of the random states
    and setting the random seed for each MP process. Megatron uses
    mpu.get_cuda_rng_tracker and mpu.model_parallel_cuda_manual_seed
    for keeping track of the random states and setting the random seeds.
    Since they are used in places outside of activation checkpointing,
    we overwrite them to maintain consistency.

    This must be called before all the calls to mpu.model_parallel_cuda_manual_seed
    '''
    num_layers = args.num_layers // args.checkpoint_num_layers
    num_layers = num_layers if args.num_layers % args.checkpoint_num_layers == 0 else num_layers + 1

    deepspeed.checkpointing.configure(
        mpu,
        partition_activations=args.partition_activations,
        contiguous_checkpointing=args.contigious_checkpointing,
        num_checkpoints=num_layers,
        checkpoint_in_cpu=args.checkpoint_in_cpu,
        synchronize=args.synchronize_each_layer,
        profile=args.profile_backward)

    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed


def _initialize_distributed():
    """Initialize torch.distributed and mpu."""
    args = get_args()

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():

        if args.rank == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...', flush=True)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        if args.rank == 0:
            print(f'> initializing torch distributed (deepspeed={args.deepspeed}) ...', flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert args.local_rank == device, \
                    'expected local-rank to be the same as rank % device-count.'
            else:
                args.local_rank = device
            torch.cuda.set_device(device)

        if args.deepspeed:
            if os.environ.get('SPOTDL_ENABLED', 'OFF') == 'ON':
                import json
                from torch.distributed.elastic.rendezvous import RendezvousParameters
                from project_pactum.rendezvous.etcd import create_rdzv_handler

                rdzv_backend = 'etcd-v2'
                rdzv_endpoint = os.environ['SPOTDL_ENDPOINT']
                run_id = os.environ['SPOTDL_RUN_ID']
                min_nodes = int(os.environ['SPOTDL_MIN_NODES'])
                max_nodes = int(os.environ['SPOTDL_MAX_NODES'])
                rdzv_configs = json.loads(os.environ['SPOTDL_RDZV_CONFIGS'])
                # rdzv_configs['last_call_timeout'] = 1
                rdzv_parameters = RendezvousParameters(
                    backend=rdzv_backend,
                    endpoint=rdzv_endpoint,
                    run_id=run_id,
                    min_nodes=min_nodes,
                    max_nodes=max_nodes,
                    **rdzv_configs,
                )
                rdzv_handler = create_rdzv_handler(rdzv_parameters)

                global_decision = rdzv_handler.get_global_decision()
                store = rdzv_handler.setup_kv_store()
                global_decision, rank, world_size, _, _ = rdzv_handler.reassign_global_rank(global_decision, store)

                deepspeed.init_distributed(
                    dist_backend=args.distributed_backend,
                    rank=rank,
                    world_size=world_size,
                    store=store)

                num_pipelines = int(os.environ['SPOTDL_NUM_PIPELINES'])
                num_stages = int(os.environ['SPOTDL_NUM_STAGES'])
                rdzv_handler.update_rank_status(rank, num_pipelines, num_stages)
                args.world_size = world_size
            else:
                deepspeed.init_distributed(dist_backend=args.distributed_backend)
        else:
            # Call the init process
            init_method = 'tcp://'
            master_ip = os.getenv('MASTER_ADDR', 'localhost')
            master_port = os.getenv('MASTER_PORT', '6000')
            init_method += master_ip + ':' + master_port
            torch.distributed.init_process_group(
                backend=args.distributed_backend,
                world_size=args.world_size, rank=args.rank,
                init_method=init_method)

    # Setup 3D topology.
    if args.pipe_parallel_size > 0:
        pp = args.pipe_parallel_size
        mp = args.model_parallel_size
        assert args.world_size % (pp * mp) == 0
        dp = args.world_size // (pp * mp)

        if os.environ.get('SPOTDL_ENABLED', 'OFF') == 'ON':
            from deepspeed.runtime.pipe.topology import PipeDataParallelTopology
            topo = PipeDataParallelTopology(num_stages, num_pipelines, global_decision)
        else:
            from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
            topo = PipeModelDataParallelTopology(num_pp=pp, num_mp=mp, num_dp=dp)

        # Offset base seeds for the interior pipeline stages.
        # TODO: adjust last stage too once IO is improved.
        stage_id = topo.get_coord(rank=torch.distributed.get_rank()).pipe
        if 0 < stage_id < topo.get_dim('pipe') - 1:
            offset = args.seed + 1138
            args.seed = offset + (stage_id * mp)
    else:
        topo = None


    # Set the model-parallel / data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print('model parallel is already initialized')
        else:
            if os.environ.get('SPOTDL_ENABLED', 'OFF') == 'ON':
                topo._attach_mpu_init_fn = functools.partial(mpu.initialize_model_parallel,
                                                             args.model_parallel_size,
                                                             reset=True)
            mpu.initialize_model_parallel(args.model_parallel_size, topology=topo)

    # Optional DeepSpeed Activation Checkpointing Features
    #
    if args.deepspeed and args.deepspeed_activation_checkpointing:
        setup_deepspeed_random_and_activation_checkpointing(args)

def _init_autoresume():
    """Set autoresume start time."""
    autoresume = get_adlr_autoresume()
    if autoresume:
        torch.distributed.barrier()
        autoresume.init()
        torch.distributed.barrier()


def _set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            mpu.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed))


def _write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    args = get_args()
    writer = get_tensorboard_writer()
    if writer:
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)))


def _initialize_mem_buffs():
    """Initialize manually allocated static memory."""
    args = get_args()

    # Initialize memory for checkpointed activations.
    if args.distribute_checkpointed_activations:
        mpu.init_checkpointed_activations_memory_buffer()
