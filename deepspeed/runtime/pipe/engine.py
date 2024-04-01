# Copyright 2019 The Microsoft DeepSpeed Team
import os
import gc
import sys
import collections
import datetime
import json
import math
import signal
import socket
import time
from collections import OrderedDict
from queue import Queue
from threading import Thread
# os.system('sudo /opt/conda/envs/pytorch/bin/pip install memory_profiler')
# from memory_profiler import profile

from colorama import Fore

from types import MethodType

import torch
from deepspeed import comm as dist
from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam

from deepspeed.utils import logger
from deepspeed.utils.timer import ThroughputTimer

from ..engine import DeepSpeedEngine, MEMORY_OPT_ALLREDUCE_SIZE
from ..utils import PartitionedTensor
from ..dataloader import RepeatingLoader

from .module import TiedLayerSpec, PipelineModule, PipelineError
from .topology import PipeDataParallelTopology
from . import p2p
from . import schedule

TARGET_ID = -2
LOG_STAGE = -2
DATA_PARALLEL_ID = -2


def is_even(number):
    return number % 2 == 0


global should_stop, debug
should_stop = False
debug = os.environ.get('SPOTDL_DEBUG', None)

def sig_handler(signum, frame):
    print(f'[Engine] Node {socket.gethostname()}, rank {os.environ.get("RANK", None)} Signal handler called with signal', signum, flush=True)
    global should_stop
    should_stop = True


class PreemptionException(Exception):
    def __init__(self, *args, dp_id=None, stage_id=None):
        super().__init__(*args)
        self.dp_id = dp_id
        self.stage_id = stage_id


class PrevStageException(PreemptionException):
    ...


class NextStageException(PreemptionException):
    ...


class AllReduceException(PreemptionException):
    ...


mem_alloced = 0
mem_cached = 0


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


class PipelineEngine(DeepSpeedEngine):
    """ A training engine hybrid pipeline, data, and model parallel training.

    This engine is created by ``deepspeed.initialize()`` when a :class:`PipelineModule`
    is provided.
    """
    ID_TO_DTYPE = [
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.float16,
        torch.bfloat16,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool
    ]
    DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}

    def __init__(self, has_bool_tensors=False, prev_state={}, prev_redundant_state={}, coordinate_decision=None, failures={}, migration_version=0, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        assert isinstance(self.module, PipelineModule), "model must base PipelineModule"

        assert self.zero_optimization_stage() < 2, "ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism"

        signal.signal(signal.SIGTERM, sig_handler)

        # We schedule the all-reduces, so disable it in super().backward()
        self.enable_backward_allreduce = False
        self.has_bool_tensors = has_bool_tensors
        self.eval_return_logits = False
        self.outputs = None

        # used to disable the pipeline all-reduce when used with 1-bit Adam/1-bit LAMB
        self.pipeline_enable_backward_allreduce = True

        if self.elasticity_enabled():
            if not self.is_elastic_model_parallel_supported():
                assert not self.elasticity_enabled(), "Elasticity is not currently supported" \
                " with pipeline parallelism."

        # pipeline step for logging
        self.log_batch_step_id = -1

        self.micro_batch_size = self.train_micro_batch_size_per_gpu()
        self.micro_batches = self.gradient_accumulation_steps()
        self.epoch = 0
        self.epoch_step = 0

        # Set Grid and Communication Groups
        self.grid = self.module._grid
        if self.grid.get_global_rank() == 0:
            logger.info(f'CONFIG: global_batch_size={self.train_batch_size()}, micro_batches={self.micro_batches} '
                        f'micro_batch_size={self.micro_batch_size}')

        self.global_rank = self.grid.get_global_rank()

        assert self.dp_world_size == self.grid.data_parallel_size
        assert self.train_batch_size() == \
            self.micro_batch_size * self.micro_batches * self.grid.data_parallel_size

        #  Set Stage Inf
        self.num_pipelines = self.grid.data_parallel_size
        self.num_stages = self.grid.pipe_parallel_size
        self.stage_id = self.grid.get_stage_id()
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

        if coordinate_decision is None:
            global_decision = self.rdzv_handler.get_global_decision()
            self.active_group_ranks = []
            for info in global_decision:
                if len(info.active_coordinates) > 0:
                    self.active_group_ranks.append(info.rank)
            global_decision = self.rdzv_handler.convert_global_decision(global_decision, self.dp_world_size // self.grid.local_world_size)
        else:
            global_decision = coordinate_decision

        self.coordinates = []
        for info in global_decision:
            if info.rank == self.global_rank:
                self.coordinates = info.active_coordinates
        self.log(f'My rank: {self.global_rank}, coordinates: {self.coordinates}, decision: {global_decision}', color='lb')

        self.global_steps = self.rdzv_handler.get_current_step()
        self.migration_version = migration_version
        if self.global_rank == 0:
            self.global_store.set('global-steps', str(self.global_steps))

        if self.global_steps == 0:
            self._broadcast_model()
        else:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(self.global_steps)

        # FIXME: inherent previous failures.
        if self.global_rank == 0:
            # self.fail_lock.acquire()
            remain_failures = {}
            for key, step in failures.items():
                if int(step) > self.global_steps:
                    remain_failures[str(key)] = step
            self.global_store.set('failures', json.dumps(remain_failures))
            # self.fail_lock.release()

        # synchronize latest model states setting
        self.enable_redundant_model_states = False
        # setup redundant model states
        # small gpt-2 model: int(4 * 1e6) # ~16MB
        self.sync_redundant_model_states_bucket_size = int(4 * 1e6)
        self.is_redundant_gradients_ready = False
        self.redundant_states_sync_interval = 1

        self.redundant_model_only = len(os.environ.get('ONDEMAND_NODE_IP', '')) > 0
        self.recv_states_from_redundant_module = os.environ.get('REDUNDANT_COMM_ONLY', '') == 'ON'
        self.save_latest = len(os.environ.get('ONDEMAND_NODE_IP', '')) > 0

        self.cpu_transfer = os.environ.get('CPU_TRANSFER', '') == 'ON'

        self.sync_redundant_grad_stream = torch.cuda.Stream()
        self._configure_redundant_comm()
        # load previous state
        # if self.is_redundant_node and len(prev_redundant_state) > 0:
        #     self.module.load_redundant_layers(prev_state, prev_redundant_state)
        #     self.load_redundant_optimizer_state(prev_state, prev_redundant_state)

        # if not self.is_first(global_decision) and self.global_steps > 0:
        if self.global_steps > 0:
            self.join = True
            self.transfer_latest_model_states(global_decision, prev_state, prev_redundant_state, need_broadcast=True)

        self.data_iterator = None
        self.batch_fn = None

        self._force_grad_boundary = False

        self.batch_timer = ThroughputTimer(batch_size=self.micro_batch_size *
                                           self.micro_batches,
                                           num_workers=self.dp_world_size,
                                           logging_fn=self.tput_log,
                                           monitor_memory=False,
                                           steps_per_output=self.steps_per_print())

        # PipelineEngine needs to handle data loading specially due to only the first
        # and last stages loading inputs/labels. We construct a sampler that uses
        if self.training_data:
            self._build_data_iter(self.training_data)

        self.is_pipe_parallel = self.grid.pipe_parallel_size > 1
        self.is_data_parallel = self.grid.data_parallel_size > 1
        self.is_model_parallel = self.grid.model_parallel_size > 1

        # Partition input/output buffers
        # XXX temporarily disable while I revert some partition hacks.
        self.is_pipe_partitioned = self.is_model_parallel
        self.is_grad_partitioned = self.is_model_parallel

        model_parameters = filter(lambda p: p.requires_grad, self.module.parameters())
        num_params = sum([p.numel() for p in model_parameters])
        unique_params = num_params
        # Subtract tied parameters if we don't own them
        if self.module.tied_comms:
            tied_params = 0
            for key, d in self.module.tied_comms.items():
                if self.global_rank != min(d['ranks']):
                    tied_params += sum(p.numel() for p in d['module'].parameters())
            unique_params -= tied_params
        params_tensor = torch.LongTensor(data=[num_params,
                                               unique_params]).to(self.device)
        dist.all_reduce(params_tensor, group=self.grid.get_model_parallel_group())
        params_tensor = params_tensor.tolist()
        total_params = params_tensor[0]
        unique_params = params_tensor[1]
        if self.grid.data_parallel_id == 0:
            logger.info(f'RANK={self.global_rank} '
                        f'STAGE={self.stage_id} '
                        f'LAYERS={self.module._local_stop - self.module._local_start} '
                        f'[{self.module._local_start}, {self.module._local_stop}) '
                        f'STAGE_PARAMS={num_params} ({num_params/1e6:0.3f}M) '
                        f'TOTAL_PARAMS={total_params} ({total_params/1e6:0.3f}M) '
                        f'UNIQUE_PARAMS={unique_params} ({unique_params/1e6:0.3f}M)')
        self.report_memory_flag = True
        del model_parameters

        # Pipeline buffers
        self.num_pipe_buffers = 0
        self.pipe_buffers = {
            'inputs' : [],   # batch input and received activations
            'labels' : [],   # labels from batch input
            'outputs' : [],  # activations
            'output_tensors' : [], # tensor object to preserve backward graph
        }
        self.pipe_recv_buf = None
        self.grad_layer = None

        self.meta_buffer = None

        self.first_output_send = True
        self.first_gradient_send = True

        #stores the loss for the current micro batch being processed
        self.loss = torch.tensor(0.0).to(self.device)

        #stores the loss for the entire batch
        self.total_loss = None
        self.agg_loss = torch.tensor(0.0, requires_grad=False).to(self.device)
        self.dp_group_loss = torch.tensor(0.0, requires_grad=False).to(self.device)

        # record lost microbatches for each epoch, reset before next epoch begins
        self.lost_samples_epoch = set() # TODO: in preemption handling logic, dump IDs of dropped samples from etcd to here
        self.num_effective_samples_epoch = 0

        if self._config.pipeline['activation_checkpoint_interval'] > 0:
            self.module.activation_checkpoint_interval = self._config.pipeline[
                'activation_checkpoint_interval']

        self.loss_model = None
        if self.is_last_stage():
            self.loss_model = self.module.loss_fn

        # remove the attention_mask by re-formulate the model
        self.has_attention_mask = False #self.module.class_name == 'GPT2ModelPipe'
        self._init_p2p_comm()

        # XXX look into timer reporting timing
        # Initialize some timers because of early weirdness.
        if self.wall_clock_breakdown():
            self.timers('forward_microstep').start()
            self.timers('forward_microstep').stop()
            self.timers('backward_microstep').start()
            self.timers('backward_microstep').stop()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward_allreduce_microstep').start()
            self.timers('backward_allreduce_microstep').stop()
            self.timers('backward_allreduce').start()
            self.timers('backward_allreduce').stop()
            self.timers('step_microstep').start()
            self.timers('step_microstep').stop()

    def set_has_attention_mask(self, value):
        assert isinstance(value, bool)
        self.has_attention_mask = value

    def _init_p2p_comm(self):
        #initialize peer-2-peer communication and allreduce groups
        # if self.is_pipe_parallel:
        #     p2p.init_process_groups(self.grid)
        p2p.init_process_groups(self.grid)

        # Initialize pipeline communicators. Just send a 0.
        if self.is_pipe_parallel:
            zero = torch.tensor(0.0).to(self.device)
            if is_even(self.stage_id):
                if not self.is_last_stage():
                    p2p.send(zero, self.next_stage)
                if not self.is_first_stage():
                    p2p.recv(zero, self.prev_stage)
            else:
                if not self.is_first_stage():
                    p2p.recv(zero, self.prev_stage)
                if not self.is_last_stage():
                    p2p.send(zero, self.next_stage)

    def _build_data_iter(self, dataset):
        self.dataset = dataset # save a pointer to build replay dataset
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=self.dp_world_size,
            rank=self.mpu.get_data_parallel_rank(),
            shuffle=False)
        # Build a loader and make it repeating.
        pipe_dataloader = self.deepspeed_io(dataset, data_sampler=sampler)
        pipe_dataloader = RepeatingLoader(pipe_dataloader)
        self.set_dataloader(pipe_dataloader)

    def _exec_reduce_tied_grads(self):
        if self.enable_redundant_model_states and self.is_redundant_node and (self.global_steps + 1) % self.redundant_states_sync_interval == 0:
            # prepare redundant gradients receiver
            self.sync_redundant_gradients()
            # update redundant model states
            self.redundant_step()

        # We need to run this first to write to self.averaged_gradients;
        # since this class turns `enable_backward_allreduce` off,
        # `self.overlapping_partition_gradients_reduce_epilogue()` defined in the DeepSpeedEngine
        # never actually runs. I suspect this is because of efficiency problems; get_flat_partition in
        # stage2.py might do something expensive; someone will have to look into that later. But
        # in the meantime, this fixes ZeRO2 + Pipelining enough to run a demo. Further profiling
        # needed to decide if it actually breaks everything.
        # (see https://github.com/EleutherAI/gpt-neox/issues/62#issuecomment-761471944)
        if self.zero_optimization_partition_gradients():
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()

        weight_group_list = self.module.get_tied_weights_and_groups()
        for weight, group in weight_group_list:
            grad = weight._hp_grad if self.bfloat16_enabled() else weight.grad
            dist.all_reduce(grad, group=group)

    def _exec_reduce_grads(self):
        self._force_grad_boundary = True

        # redundant node finish synchronzie redundant gradients
        if self.enable_redundant_model_states and self.is_redundant_node:
            # start several redundant gradients receiver during ending phase
            if self.is_redundant_gradients_ready:
                self.recv_redundant_gradients(self.redundant_gradients_recv_num['ending_phase'])
            # if (self.global_steps + 1) % self.redundant_states_sync_interval == 0:
            #     self.sync_redundant_gradients_wait()
            self.sync_redundant_gradients_wait()

        def reduce_handler():
            if self.pipeline_enable_backward_allreduce:
                if self.bfloat16_enabled():
                    if self.zero_optimization_stage() == 0:
                        self._bf16_reduce_grads()
                    else:
                        assert self.zero_optimization_stage() == 1, "only bf16 + z1 are supported"
                        raise NotImplementedError()
                else:
                    self.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
        try:
            reduce_handler()
        except Exception as e:
            raise
            failed_dp_id = None
            info = f'---- STEP {self.global_steps}, stage_id={self.stage_id} ALL-REDUCE GROUP FAILED DURING COMM. USING FALLBACK'
            for rank in self.grid.dp_group:
                if int(self.global_store.get(str(rank))) == 1:
                    failed_dp_id = self.grid._topo.get_coord(rank).data
                    info = f'---- STEP {self.global_steps}, RANK {rank} (pipe={self.stage_id}, data={failed_dp_id})'
                    info += f' IN ALL-REDUCE GROUP FAILED DURING COMM. USING FALLBACK'
            self.log(info, color='b')

            self._force_grad_boundary = False
            raise AllReduceException(e, dp_id=failed_dp_id, stage_id=self.stage_id)

        self._force_grad_boundary = False

    def _bf16_reduce_grads(self):
        # Make our own list of gradients from the optimizer's FP32 grads
        grads = []
        self.buffered_allreduce_fallback(grads=self.optimizer.get_grads_for_reduction(),
                                         elements_per_buffer=MEMORY_OPT_ALLREDUCE_SIZE)

    def _reserve_pipe_buffers(self, num_buffers):
        """Ensure that each pipeline buffer has at least ``num_buffers`` slots.

        This method only reserves slots and does not allocate tensors.

        Args:
            num_buffers (int): The number of buffers to reserve.
        """
        if self.num_pipe_buffers >= num_buffers:
            return

        num_added = num_buffers - self.num_pipe_buffers
        for key in self.pipe_buffers:
            self.pipe_buffers[key].extend([None] * num_added)
        self.num_pipe_buffers = num_buffers

    def reset_activation_shape(self):
        """Reset the buffers when the shape of activation and gradient change.
        For example, for curriculum learning that changes the seqlen of each
        sample, we need to call this whenever the seqlen is going to change.
        """
        self.first_output_send = True
        self.pipe_recv_buf = None
        self.grad_layer = None
        self.meta_buffer = None

    def prepare_redundant_recv_nums(self, bucket_num):
        # for gpt-2 small: 5, 4
        self.redundant_warmup_phase_num = min(2, bucket_num)
        self.redundant_ending_phase_num = min(2, bucket_num - self.redundant_warmup_phase_num)

        total = bucket_num - (self.redundant_ending_phase_num + self.redundant_warmup_phase_num) * self.redundant_states_sync_interval
        part = total // (self.micro_batches * self.redundant_states_sync_interval)
        assert part >= 0
        recv_num_list = [part] * self.micro_batches * self.redundant_states_sync_interval
        for i in range(total % (self.micro_batches * self.redundant_states_sync_interval)):
            recv_num_list[-i-1] += 1

        forward_recv_num_list = []
        for i in range(len(recv_num_list)):
            # forward_recv_num_list.append(1)
            # recv_num_list[i] -= 1
            forward_recv_num_list.append(0)

        self.redundant_gradients_recv_num = {
            'ending_phase': self.redundant_ending_phase_num,
            'warmup_phase': self.redundant_warmup_phase_num,
            'steady_phase': recv_num_list,
            'steady_forward_phase': forward_recv_num_list,
        }

        # for stage_id, stage_bucket_num in self.stage_to_bucket_num.items():
        #     self.prepare_redundant_send_nums(stage_bucket_num, stage_id)
        #     self.stage_recv_nums[stage_id] = self.redundant_gradients_send_num.copy()

    def prepare_redundant_send_nums(self, bucket_num, stage_id=None):
        self.redundant_gradients_send_num = self.stage_recv_num.copy()

    def _configure_redundant_comm(self):
        self.is_redundant_node = False
        self.is_redundant_sender = False
        if self.enable_redundant_model_states:
            # currently only data_parallel_rank=0 nodes will sync
            self.is_redundant_node = (self.stage_id == self.num_stages - 1 and self.grid.data_parallel_id == 0)
            self.is_redundant_sender = (self.stage_id != self.num_stages - 1 and self.grid.data_parallel_id == 0)
            self.recv_states_from_redundant_module = os.environ.get('REDUNDANT_COMM_ONLY', '') == 'ON'

        if self.redundant_model_only:
            self.is_redundant_node = (self.stage_id == self.num_stages - 1 and self.grid.data_parallel_id == 0)

        if self.is_redundant_node:
            self.log(f'Rank {self.global_rank} is a redundant node with coordinates {self.coordinates}, begin to build redundant layers...')
            self.module.build_redundant_layers()
            # self.build_redundant_optimizer()
            self.sync_redundant_gradients_handlers = {}
            self.redundant_gradients_receivers = []
        else:
            self.redundant_gradients_senders = []
            self.sync_redundant_gradients_handlers = []

        if self.is_redundant_node and not self.redundant_model_only:
            bucket_num = 0
            self.stage_to_bucket_num = {}
            for stage_id in range(self.num_stages - 1):
                stage_bucekt_num = 0
                bucket_size = 0
                for param in self.module.redundant_parameters(stage_id):
                    if param.requires_grad:
                        bucket_size += param.numel()

                    if bucket_size >= self.sync_redundant_model_states_bucket_size:
                        bucket_num += 1
                        bucket_size = 0
                        stage_bucekt_num += 1
                if bucket_size > 0:
                    bucket_num += 1
                    stage_bucekt_num += 1

                self.stage_to_bucket_num[stage_id] = stage_bucekt_num

            self.prepare_redundant_recv_nums(bucket_num)

            self.log(f'Redundant Recv Number: {self.redundant_gradients_recv_num}', color='y')

            # recv num in each interval
            recv_num_list = self.redundant_gradients_recv_num['steady_phase']
            base = self.redundant_ending_phase_num + self.redundant_warmup_phase_num
            interval_num = [base] * self.redundant_states_sync_interval
            for i in range(self.redundant_states_sync_interval):
                interval_num[i] += sum(recv_num_list[i * self.micro_batches: (i+1) * self.micro_batches])

            self.stage_recv_nums = {stage_id: [0] * self.redundant_states_sync_interval for stage_id in range(self.num_stages - 1)}
            stage_to_bucket_num = self.stage_to_bucket_num.copy()
            self.log(f'total bucket num: {bucket_num}, stage_to_bucket_num: {sum(list(stage_to_bucket_num.values()))}, list: {stage_to_bucket_num}', color='y')
            self.log(f'interval_num: {interval_num}, sum: {sum(interval_num)}', color='y')
            stage_id = 0
            for i in range(self.redundant_states_sync_interval):
                total_num = interval_num[i]
                while total_num > 0:
                    if stage_to_bucket_num[stage_id] > 0:
                        self.stage_recv_nums[stage_id][i] += 1
                        stage_to_bucket_num[stage_id] -= 1
                        total_num -= 1
                    stage_id = (stage_id + 1) % (self.num_stages - 1)

            distribute_tensor = []
            for stage_id in range(self.num_stages - 1):
                distribute_tensor += self.stage_recv_nums[stage_id]
            distribute_tensor = torch.tensor(distribute_tensor, dtype=torch.int32).cuda()
            dist.all_reduce(distribute_tensor)
        elif self.enable_redundant_model_states:
            recv_num_list = [0] * (self.redundant_states_sync_interval * (self.num_stages - 1))
            recv_num_list = torch.tensor(recv_num_list, dtype=torch.int32).cuda()
            dist.all_reduce(recv_num_list)
            my_recv_num_list = recv_num_list.cpu().numpy().tolist()[self.stage_id*self.redundant_states_sync_interval: (self.stage_id+1)*self.redundant_states_sync_interval]
            self.stage_recv_num = my_recv_num_list

        if 0 in self.grid.get_active_ranks():
            self.is_log_node = (self.global_rank == 0)
        elif self.enable_redundant_model_states:
            self.is_log_node = self.is_redundant_node
        else:
            self.is_log_node = (self.grid.data_parallel_id == 0 and self.stage_id == 0)

    def exit_entrypoint(self, exitcode):
        torch.cuda.synchronize()
        self.rdzv_handler.stop_keep_alive()
        if exitcode == 13:
            self.log(f'Rank {self.global_rank} ({socket.gethostname()}) is preempted', color='r')
        if exitcode == 125 and self.grid.local_rank == 0:
            self.rdzv_handler.set_unused(self.group_rank)
        sys.exit(exitcode)

    def is_first(self, globlal_decisions):
        '''
            Check if this is the first initialization of the cluster
            or if there are existing nodes which have more up to date
            state
        '''
        for info in globlal_decisions:
            if len(info.previous_coordinates) != 0:
                return False

        return True

    def find_prev_strategy(self):
        prev_strategy = self.rdzv_handler.get_prev_strategy()
        previous_num_pipelines, previous_num_stages = prev_strategy
        previous_num_pipelines = previous_num_pipelines * self.grid.local_world_size
        return previous_num_pipelines, previous_num_stages

    def add_optimizer_state(self, l_id, lyr, state, redundant=False):
        optim_state = state[l_id][1]
        if hasattr(lyr, 'parameters'):
            with torch.no_grad():
                state_idx = 0
                for j, (name, p) in enumerate(lyr.named_parameters()):
                    if not ('weight' in name or 'bias' in name):
                        continue
                    if redundant:
                        for key, value in optim_state[state_idx].items():
                            if key in self.redundant_optimizer.state[p]:
                                self.redundant_optimizer.state[p][key].copy_(value)
                            else:
                                self.redundant_optimizer.state[p][key] = value.clone().cpu().detach()
                        # state = {}
                        #     if torch.is_tensor(value) and value.is_cuda:
                        #         state[key] = value.clone().cpu().detach()
                        #     else:
                        #         state[key] = value.clone().detach()
                        # self.redundant_optimizer.state[p] = state
                    else:
                        for key, value in optim_state[state_idx].items():
                            if key in self.optimizer.state[p]:
                                self.optimizer.state[p][key].copy_(value)
                            else:
                                self.optimizer.state[p][key] = value.clone().detach()
                    state_idx += 1

    def load_optimizer_state(self, recvd_state, prev_state={}):
        bucket = self.module.func_buckets[self.stage_id]
        if self.fp16_enabled():
            fp16_params, fp32_states = [], {}
            for i, param_group in enumerate(self.optimizer.fp16_groups):
                opt_states_flat = self.optimizer.state[self.optimizer.fp32_groups_flat[i]]
                for key, flat_state in opt_states_flat.items():
                    if key not in fp32_states:
                        fp32_states[key] = []
                    fp32_states[key].extend(self.unflatten(flat_state, param_group))

                flat_fp16_param = self.flatten(param_group)
                self.optimizer.fp32_groups_flat[i].data.copy_(flat_fp16_param.float())
                self.optimizer.fp16_groups_flat[i].data.copy_(flat_fp16_param)

            for i, param_group in enumerate(self.optimizer.fp16_groups):
                for param in param_group:
                    fp16_params.append(param.data_ptr())

            stage_part_start = self.module.parts[self.stage_id]
            for i, module in enumerate(bucket):
                l_id = stage_part_start + i
                layer_spec = self.module._layer_specs[l_id]
                if isinstance(layer_spec, TiedLayerSpec):
                    module = self.module.tied_modules[layer_spec.key]
                if not hasattr(module, 'state_dict') or not hasattr(module, 'parameters'):
                    continue

                if l_id in prev_state:
                    optim_state = prev_state[l_id][1]
                elif l_id in recvd_state:
                    optim_state = recvd_state[l_id][1]
                else:
                    continue
                    if self.enable_redundant_model_states or self.redundant_model_only:
                        raise ValueError(f'No optimizer state for layer {l_id}')

                for j, param in enumerate(module.parameters()):
                    idx = fp16_params.index(param.data_ptr())
                    for state_key in fp32_states:
                        fp32_states[state_key][idx].data.copy_(optim_state[j][state_key])

                if l_id in prev_state:
                    del prev_state[l_id]
                elif l_id in recvd_state:
                    del recvd_state[l_id]
                del optim_state
            start_id = 0
            for i, param_group in enumerate(self.optimizer.fp16_groups):
                opt_states_flat = self.optimizer.state[self.optimizer.fp32_groups_flat[i]]
                for key, flat_state in opt_states_flat.items():
                    new_state = self.flatten(fp32_states[key][start_id:start_id + len(param_group)]).cuda().float()
                    flat_state.data.copy_(new_state)
                start_id += len(param_group)
            del fp16_params, fp32_states, flat_state
        else:
            for i, l_id in enumerate(range(self.module.parts[self.stage_id], self.module.parts[self.stage_id + 1])):
                layer_spec = self.module._layer_specs[l_id]
                if isinstance(layer_spec, TiedLayerSpec):
                    module = self.tied_modules[layer_spec.key]
                else:
                    module = bucket[i]

                if l_id in prev_state:
                    self.add_optimizer_state(l_id, module, prev_state)
                    continue
                elif l_id in recvd_state:
                    self.add_optimizer_state(l_id, module, recvd_state)
                    continue
                else:
                    if self.grid.data_parallel_id == 0:
                        self.log('No optimizer state for layer {}'.format(l_id), color='y')

    def load_redundant_optimizer_state(self, prev_state, prev_redundant_state):
        for stage_id in range(self.num_stages - 1):
            bucket = self.module.func_buckets[stage_id]
            for i, l_id in enumerate(range(self.module.parts[stage_id], self.module.parts[stage_id + 1])):
                layer_spec = self.module._layer_specs[l_id]
                if isinstance(layer_spec, TiedLayerSpec):
                    module = self.tied_modules[layer_spec.key]
                    continue
                else:
                    module = bucket[i]

                if l_id in prev_state:
                    self.add_optimizer_state(l_id, module, prev_state, redundant=True)
                    continue

                if l_id in prev_redundant_state:
                    self.add_optimizer_state(l_id, module, prev_redundant_state, redundant=True)
                    continue

    def send_layers(self, dst_rank, layer_idxs, state, broadcast=False):
        """ Move a number of layers from this rank to dst rank

        Args:
            dst_rank (int): the destination rank that is receiving layers
            layer_idxs (list): The global ids of the layers
            state (dict): Dict of { layer_idx: }
        """
        print(Fore.LIGHTYELLOW_EX, f"TRANSFERRING {', '.join([str(idx) for idx in layer_idxs])} TO", dst_rank, Fore.RESET)
        layer_bucket = []
        optim_bucket = []
        for idx in layer_idxs:
            layer_state = state[idx][0]
            optim_state = state[idx][1]

            if self.redundant_model_only and self.is_redundant_node and not self.save_latest:
                for key, param_tensor in layer_state.items():
                    layer_bucket.append(param_tensor.clone().detach())
                    for _ in range(2):
                        optim_bucket.append(param_tensor.clone().detach())
            else:
                for key, param_tensor in layer_state.items():
                    if 'running_mean' in key or 'running_var' in key:
                        running_tensor = param_tensor

                    if 'num_batches_tracked' in key:
                        # layer_bucket.append(torch.ones_like(running_tensor))
                        layer_bucket.append(param_tensor.clone().detach().float())
                    else:
                        layer_bucket.append(param_tensor)

                for optim_dict in optim_state:
                    for key, tensor_value in optim_dict.items():
                        optim_bucket.append(tensor_value)

        # check device
        cpu_size, cuda_size = 0, 0
        for tensor in layer_bucket:
            if tensor.is_cuda:
                cuda_size += tensor.numel()
            else:
                cpu_size += tensor.numel()
        if cpu_size > 0 and cuda_size > 0 and not self.cpu_transfer:
            for i in range(len(layer_bucket)):
                layer_bucket[i] = layer_bucket[i].cuda()
            for i in range(len(optim_bucket)):
                optim_bucket[i] = optim_bucket[i].cuda()
        elif cuda_size > 0 and self.cpu_transfer:
            for i in range(len(layer_bucket)):
                layer_bucket[i] = layer_bucket[i].cpu()
            for i in range(len(optim_bucket)):
                optim_bucket[i] = optim_bucket[i].cpu()

        if len(layer_bucket) <= 0:
            self.log(f'No layers to send to {dst_rank}', color='lg')
            return

        if self.cpu_transfer:
            layer_tensor = self.flatten(layer_bucket).cpu().float().detach()
            optim_tensor = self.flatten(optim_bucket).cpu().float().detach()
        else:
            layer_tensor = self.flatten(layer_bucket).cuda().float().detach()
            optim_tensor = self.flatten(optim_bucket).cuda().float().detach()

        self.log(f'SIZE OF LAYER TENSOR {layer_tensor.size()} ({layer_tensor.dtype}, {layer_tensor.device}, {layer_tensor.requires_grad})', color='lg')
        self.log(f'SIZE OF OPTIM TENSOR {optim_tensor.size()} ({optim_tensor.dtype}, {optim_tensor.device}, {optim_tensor.requires_grad})', color='lg')
        #group = None if layer_tensor.is_cuda else self.gloo_pg

        ## Send the layers and optimizer state
        if dst_rank != self.global_rank:
            if self.cpu_transfer:
                dist.send(layer_tensor, dst=dst_rank, group=self.grid.gloo_pg)
                dist.send(optim_tensor, dst=dst_rank, group=self.grid.gloo_pg)
            else:
                dist.send(layer_tensor, dst=dst_rank) #, group=group)
                dist.send(optim_tensor, dst=dst_rank) #, group=group)

        # if broadcast:
        #     broadcast_root = self.global_rank
        #     if self.cpu_transfer:
        #         dist.broadcast(layer_tensor, src=broadcast_root, group=self.grid.dp_proc_cpu_group)
        #         dist.broadcast(optim_tensor, src=broadcast_root, group=self.grid.dp_proc_cpu_group)
        #     else:
        #         dist.broadcast(layer_tensor, src=broadcast_root, group=self.grid.get_data_parallel_group())
        #         dist.broadcast(optim_tensor, src=broadcast_root, group=self.grid.get_data_parallel_group())

        del layer_tensor, optim_tensor, layer_bucket, optim_bucket

    def recv_layers(self, src_rank, layer_idxs, broadcast_root=None, stage_id=None):
        """ Receive a set of layers from rank src
        Args:
            layer_idxs (list): The global ids of the layers to move
            src_rank (int): The source rank that is sending the layers
        """
        ## JOHN: Start with a simple implementation but hope to eventually use the
        ## same bucketing technique used in the all-reduce to speed up
        layer_bucket = []
        optim_state_bucket = []
        if stage_id is None:
            stage_id = self.stage_id

        print(Fore.LIGHTYELLOW_EX, f"RECEIVING {', '.join([str(idx) for idx in layer_idxs])} FROM", src_rank, Fore.RESET)
        layer_state_dicts = []
        for idx in layer_idxs:
            local_start, local_stop = self.module.parts[stage_id], self.module.parts[stage_id + 1]
            if idx < local_start or idx >= local_stop:
                self.log(f'{idx} is not in this rank (rank parts: {local_start}:{local_stop})', color='r')
                assert False
            local_id = idx - local_start

            layer_spec = self.module._layer_specs[idx]
            if isinstance(layer_spec, TiedLayerSpec):
                module = self.module.tied_modules[layer_spec.key]
            else:
                module = self.module.func_buckets[stage_id][local_id]

            if not hasattr(module, 'parameters'):
                layer_state_dicts.append({})
                continue
            layer_state_dicts.append(module.state_dict())

            for key, p in layer_state_dicts[-1].items():
                if 'running_mean' in key or 'running_var' in key:
                    running_tensor = p

                if 'num_batches_tracked' in key:
                    # layer_bucket.append(torch.ones_like(running_tensor))
                    layer_bucket.append(p.clone().detach().float())
                else:
                    if self.cpu_transfer:
                        layer_bucket.append(torch.ones_like(p, dtype=torch.float32, device='cpu'))
                    else:
                        layer_bucket.append(torch.ones_like(p))

            for p in module.parameters():
                ## Hardcoded for the FusedAdam optimizer which has two optim state
                ## tensors for every parameter
                for _ in range(2):
                    if self.cpu_transfer:
                        optim_state_bucket.append(torch.ones_like(p, dtype=torch.float32, device='cpu'))
                    else:
                        optim_state_bucket.append(torch.ones_like(p))

        if len(layer_bucket) <= 0:
            self.log(f'No layers to receive from {src_rank}', color='lg')
            return {}

        if self.cpu_transfer:
            layer_tensor = self.flatten(layer_bucket).cpu().float()
            optim_tensor = self.flatten(optim_state_bucket).cpu().float()
        else:
            layer_tensor = self.flatten(layer_bucket).cuda().float()
            optim_tensor = self.flatten(optim_state_bucket).cuda().float()

        self.log(f'SIZE OF LAYER TENSOR {layer_tensor.size()} ({layer_tensor.dtype}, {layer_tensor.device}, {layer_tensor.requires_grad})', color='lg')
        self.log(f'SIZE OF OPTIM TENSOR {optim_tensor.size()} ({optim_tensor.dtype}, {optim_tensor.device}, {optim_tensor.requires_grad})', color='lg')
        #group = None if layer_tensor.is_cuda else self.gloo_pg

        if broadcast_root is None or broadcast_root == self.global_rank:
            if self.cpu_transfer:
                dist.recv(layer_tensor, src=src_rank, group=self.grid.gloo_pg)
                dist.recv(optim_tensor, src=src_rank, group=self.grid.gloo_pg)
            else:
                dist.recv(layer_tensor, src=src_rank) #, group=group)
                dist.recv(optim_tensor, src=src_rank) #, group=group)

        # if broadcast_root is not None:
        #     if self.cpu_transfer:
        #         dist.broadcast(layer_tensor, src=broadcast_root, group=self.grid.dp_proc_cpu_group)
        #         dist.broadcast(optim_tensor, src=broadcast_root, group=self.grid.dp_proc_cpu_group)
        #     else:
        #         dist.broadcast(layer_tensor, src=broadcast_root, group=self.grid.get_data_parallel_group())
        #         dist.broadcast(optim_tensor, src=broadcast_root, group=self.grid.get_data_parallel_group())

        recvd_state = {}

        index = 0
        received_lsds = self.unflatten(layer_tensor, layer_bucket)
        for i in range(len(layer_state_dicts)):
            sd = layer_state_dicts[i]
            if len(sd) == 0:
                self.log(f'No state dict for layer {i}', color='lg')
                recvd_state[layer_idxs[i]] = [{}, []]
                continue

            for k in sd:
                sd[k] = received_lsds[index]
                index += 1

            recvd_state[layer_idxs[i]] = [sd, []]

        received_optim_tensors = self.unflatten(optim_tensor, optim_state_bucket)
        index = 0
        state_keys = ['exp_avg', 'exp_avg_sq']
        assert len(received_optim_tensors) % len(state_keys) == 0
        for i in range(len(layer_state_dicts)):
            sd = layer_state_dicts[i]
            if len(sd) == 0:
                continue

            optim_state_dicts_list = []
            for key, v in sd.items():
                optim_state_dict = {}
                if 'weight' in key or 'bias' in key:
                    for k in state_keys:
                        assert v.size() == received_optim_tensors[index].size()
                        optim_state_dict[k] = received_optim_tensors[index]
                        index += 1

                    optim_state_dicts_list.append(optim_state_dict)

            recvd_state[layer_idxs[i]][1] = optim_state_dicts_list

        del layer_tensor, optim_tensor, layer_bucket, optim_state_bucket, layer_state_dicts

        return recvd_state

    def transfer_layers(self, recv_decisions, send_decisions={}, prev_state={}):
        received_state = {}

        ## Implement sync transfer protocol that I just though of
        my_send_decicions = send_decisions[self.global_rank] if self.global_rank in send_decisions else {}
        my_recv_decisions = recv_decisions[self.global_rank] if self.global_rank in recv_decisions else {}

        # re-order send and recv decisions
        if self.cpu_transfer:
            all_transfers = []
            volume_dict = {}
            for src in send_decisions:
                for dst in send_decisions[src]:
                    volume_dict[(src, dst)] = len(send_decisions[src][dst])
                    all_transfers.append((src, dst))

            all_transfers = sorted(all_transfers, key=lambda x: (volume_dict[x], x[0], x[1]), reverse=True)
            first_transfer, remain_transfer = [], []
            busy_ranks = set()
            while len(all_transfers) > 0:
                src, dst = all_transfers.pop(0)
                if src in busy_ranks or dst in busy_ranks:
                    remain_transfer.append((src, dst))
                else:
                    first_transfer.append((src, dst))
                    busy_ranks.add(src)
                    busy_ranks.add(dst)

            for src, dst in first_transfer + remain_transfer:
                if src == self.global_rank:
                    layer_idxs = my_send_decicions[dst]
                    if len(layer_idxs) > 0:
                        self.send_layers(dst, sorted(layer_idxs), prev_state)
                elif dst == self.global_rank:
                    layer_idxs = my_recv_decisions[src]
                    if len(layer_idxs) > 0:
                        received_state.update(self.recv_layers(src, sorted(layer_idxs)))
        else:
            for rank in range(self.world_size):
                if rank == self.global_rank:
                    for dst_rank, layer_idxs in my_send_decicions.items():
                        if isinstance(layer_idxs, dict):
                            self.send_layers(dst_rank, sorted(layer_idxs['layers']), prev_state)
                        else:
                            if len(layer_idxs) == 0:
                                continue

                            self.send_layers(dst_rank, sorted(layer_idxs), prev_state)
                else:
                    if rank in my_recv_decisions:
                        src_rank = rank
                        layer_idxs = my_recv_decisions[rank]
                        if isinstance(layer_idxs, dict):
                            received_state.update(self.recv_layers(src_rank, sorted(layer_idxs['layers']), stage_id=layer_idxs['stage_id']))
                        else:
                            received_state.update(self.recv_layers(src_rank, sorted(layer_idxs)))

        return received_state

    def transfer_redundant_layers(self, redundant_send_decisions, prev_redundant_state={}):
        received_redundant_state = {}

        # self.log(f'rank {self.global_rank} transfer_redundant_layers, decisions: {redundant_send_decisions}', color='lb')
        # self.log(f'rank {self.global_rank} layers: {list(prev_redundant_state.keys())}', color='lb')

        for decision in redundant_send_decisions:
            layer_idxs = decision['layer_idxs']
            src_rank = decision['source_node']
            dst_rank = decision['target_node']
            ranks = decision['ranks']

            if self.is_redundant_node:
                broadcast = (self.global_rank == dst_rank)
                self.send_layers(dst_rank, sorted(layer_idxs), prev_redundant_state, broadcast=broadcast)
            else:
                if self.global_rank in ranks:
                    recvd_state = self.recv_layers(src_rank, sorted(layer_idxs), broadcast_root=dst_rank)
                    received_redundant_state.update(recvd_state)

        return received_redundant_state

    def broadcast_model_states(self, group, root):
        for p in self.module.parameters():
            dist.broadcast(p, src=root, group=group)
            for key, optim_tensor in self.optimizer.state[p].items():
                dist.broadcast(optim_tensor, src=root, group=group)

    def remove_param_optim_state(self, param, redundant=False):
        optimizer = self.redundant_optimizer if redundant else self.optimizer
        del optimizer.state[param]

        for param_group in optimizer.param_groups:
            for i in range(len(param_group['params'])):
                t = param_group['params'][i]
                ## Annoying way to find the parameters in the param groups
                ## Have to compare the underlying data ptr
                if t.data_ptr() == param.data_ptr():
                    del param_group['params'][i]
                    break

    def get_layer_and_optim_state(self, stage_part_start, func_bucket, delete_state, redundant=False):
        bucket_state = {}
        if self.fp16_enabled() and not redundant:
            fp16_params, fp32_params, fp32_states = [], [], {}
            for i, param_group in enumerate(self.optimizer.fp16_groups):
                fp32_param_groups = self.unflatten(self.optimizer.fp32_groups_flat[i], param_group)
                fp32_params.extend(fp32_param_groups)

                opt_states_flat = self.optimizer.state[self.optimizer.fp32_groups_flat[i]]
                for key, flat_state in opt_states_flat.items():
                    if key not in fp32_states:
                        fp32_states[key] = []
                    fp32_states[key].extend(self.unflatten(flat_state, param_group))

            for i, param_group in enumerate(self.optimizer.fp16_groups):
                for param in param_group:
                    fp16_params.append(param.data_ptr())

            for i, module in enumerate(func_bucket):
                layer_spec = self.module._layer_specs[stage_part_start + i]
                if isinstance(layer_spec, TiedLayerSpec):
                    module = self.module.tied_modules[layer_spec.key]
                    if redundant:
                        continue

                if not hasattr(module, 'state_dict') or not hasattr(module, 'parameters'):
                    bucket_state[stage_part_start + i] = ({}, [])
                    continue

                module_state_dict = OrderedDict()
                module_optim_state = []
                for key, param in module.state_dict().items():
                    if 'weight' in key or 'bias' in key:
                        idx = fp16_params.index(param.data_ptr())
                        if self.cpu_transfer:
                            module_state_dict[key] = fp32_params[idx].detach().cpu()
                        else:
                            module_state_dict[key] = fp32_params[idx].detach()
                        opt_state = {}
                        for state_key in fp32_states:
                            if self.cpu_transfer:
                                opt_state[state_key] = fp32_states[state_key][idx].detach().cpu()
                            else:
                                opt_state[state_key] = fp32_states[state_key][idx].detach()
                        module_optim_state.append(opt_state)
                    else:
                        if self.cpu_transfer:
                            module_state_dict[key] = param.cpu().detach()
                        else:
                            module_state_dict[key] = param.detach()

                bucket_state[stage_part_start + i] = (module_state_dict, module_optim_state)
        else:
            for i, module in enumerate(func_bucket):
                layer_spec = self.module._layer_specs[stage_part_start + i]
                if isinstance(layer_spec, TiedLayerSpec):
                    module = self.module.tied_modules[layer_spec.key]
                    if redundant:
                        continue

                if not hasattr(module, 'state_dict'):
                    bucket_state[stage_part_start + i] = ({}, [])
                    continue

                module_optim_state = []
                if hasattr(module, 'parameters'):
                    for name, p in module.named_parameters():
                        if not ('weight' in name or 'bias' in name):
                            continue
                        if redundant:
                            module_optim_state.append(self.redundant_optimizer.state[p])
                        else:
                            module_optim_state.append(self.optimizer.state[p])
                        if delete_state:
                            self.remove_param_optim_state(p, redundant=redundant)
                bucket_state[stage_part_start + i] = (module.state_dict(), module_optim_state)

        return bucket_state

    def get_model_state(self, delete_state=True):
        model_state = {}
        stage_part_start = self.module.parts[self.stage_id]
        stage_partition = self.module.func_buckets.get(self.stage_id, None)
        model_state.update(self.get_layer_and_optim_state(stage_part_start, stage_partition, delete_state))
        return model_state

    def get_redundant_states(self, prev_state, delete_state=True):
        if not self.is_redundant_node:
            return {}
        model_state = {}
        prev_num_stages = len(self.module.parts) - 1
        for stage_id in range(prev_num_stages - 1):
            stage_part_start = self.module.parts[stage_id]
            stage_partition = self.module.func_buckets[stage_id]
            model_state.update(self.get_layer_and_optim_state(stage_part_start, stage_partition, delete_state, redundant=True))

            # add additional TiedModule states
            stage_part_end = self.module.parts[stage_id + 1]
            for i, l_id in enumerate(range(stage_part_start, stage_part_end)):
                layer_spec = self.module._layer_specs[l_id]
                if isinstance(layer_spec, TiedLayerSpec):
                    module = self.module.tied_modules[layer_spec.key]
                    if l_id in model_state:
                        continue

                    local_start = self.module.parts[self.stage_id]
                    local_end = self.module.parts[self.stage_id + 1]
                    for local_id in range(local_start, local_end):
                        local_layer_spec = self.module._layer_specs[local_id]
                        if isinstance(local_layer_spec, TiedLayerSpec) and local_layer_spec.key == layer_spec.key:
                            model_state[l_id] = prev_state[local_id]

        return model_state

    def get_recv_decisions(self, old_parts, new_parts, global_decisions):
        prev_stage_to_rank, comm_counts = {}, {}
        for info in global_decisions:
            for coord in info.previous_coordinates:
                if coord[1] not in prev_stage_to_rank:
                    prev_stage_to_rank[coord[1]] = []
                prev_stage_to_rank[coord[1]].append(info.rank)
            comm_counts[info.rank] = 0
        # self.log(f'prev_stage_to_rank: {prev_stage_to_rank}, decision: {global_decisions}', color='g')

        recv_decisions = {}
        redundant_recv_decisions = {}
        for info in global_decisions:
            if len(info.active_coordinates) == 0 or info.active_coordinates[0][0] != 0:
                continue

            rank = info.rank
            rank_recv_decisions = {}
            my_stage = info.active_coordinates[0][1]
            needed_layers = set(range(new_parts[my_stage], new_parts[my_stage + 1]))

            # if len(info.previous_coordinates) != 0:
            #     prev_stages = [s_id for dp_id, s_id in info.previous_coordinates]
            #     prev_partition = set(range(old_parts[min(prev_stages)], old_parts[max(prev_stages) + 1]))
            #     needed_layers.difference_update(prev_partition)

            # select stages to receive from
            for stage_id in range(len(old_parts) - 1):
                if len(needed_layers) == 0:
                    break
                if self.recv_states_from_redundant_module:
                    break
                part_start = old_parts[stage_id]
                part_end = old_parts[stage_id + 1]
                stage_prev_parts = set(range(part_start, part_end))

                intersect = needed_layers.intersection(stage_prev_parts)
                if len(intersect) == 0:
                    continue

                # select the best node in that stage
                if stage_id not in prev_stage_to_rank or len(prev_stage_to_rank[stage_id]) == 0:
                    # FIXME: no node in that stage
                    continue

                src_rank = min([rank for rank in prev_stage_to_rank[stage_id]], key=lambda x: comm_counts[x])
                if src_rank == rank:
                    continue
                comm_counts[src_rank] += len(intersect)
                comm_counts[rank] += len(intersect)
                rank_recv_decisions[src_rank] = intersect
                needed_layers.difference_update(intersect)

            recv_decisions[rank] = rank_recv_decisions
            if self.redundant_model_only or self.enable_redundant_model_states:
                if info.active_coordinates[0][0] == 0 and my_stage == len(new_parts) - 2:
                    redundant_recv_decisions['redundant_node'] = rank
                    is_redundant_node = True
                else:
                    is_redundant_node = False
                if len(needed_layers) != 0 and not is_redundant_node:
                    redundant_recv_decisions[rank] = {
                        'needed_layers': needed_layers,
                        'coordinate': info.active_coordinates[0]
                    }
        return recv_decisions, redundant_recv_decisions

    def get_redundant_send_decisions(self, redundant_recv_decisions):
        redundant_node = redundant_recv_decisions.pop('redundant_node', None)
        if redundant_node is None:
            return {}

        needed_layers_to_ranks, rank_to_coordinate = {}, {}
        for rank, item in redundant_recv_decisions.items():
            needed_layers = item['needed_layers']
            coordinate = item['coordinate']
            key = tuple(sorted(list(needed_layers)))
            if key not in needed_layers_to_ranks:
                needed_layers_to_ranks[key] = []
            needed_layers_to_ranks[key].append(rank)
            rank_to_coordinate[rank] = coordinate

        # build send decisions
        redundant_send_decisions = []
        for needed_layers, ranks in needed_layers_to_ranks.items():
            if redundant_node in ranks:
                target_node = redundant_node
            else:
                target_node = ranks[0]

            # these nodes should in the same stage
            stages = set()
            for rank in ranks:
                stages.add(rank_to_coordinate[rank][1])
            # assert len(stages) == 1
            # assert self.dp_world_size == len(ranks)

            decision = {
                'layer_idxs': needed_layers,
                'source_node': redundant_node,
                'target_node': target_node,
                'ranks': ranks,
            }
            redundant_send_decisions.append(decision)
        return redundant_send_decisions

    def get_send_decisions(self, recv_decisions):
        send_decisions = {}
        for recving_rank, recv_info in recv_decisions.items():
            for sending_rank, layers in recv_info.items():
                if sending_rank not in send_decisions:
                    send_decisions[sending_rank] = {}

                send_decisions[sending_rank][recving_rank] = layers

        return send_decisions

    def show_param_norm(self, prefix=''):
        pass
        # for name, param in self.module.named_parameters():
        #     if len(self.optimizer.state[param]) > 0:
        #         self.log(f'{prefix} opt step: {self.optimizer.param_groups[0]["step"]}, param name: {name}, {param.norm()}, exp: {self.optimizer.state[param]["exp_avg"].norm()}, {self.optimizer.state[param]["exp_avg_sq"].norm()}', color='lb')
        #     else:
        #         self.log(f'{prefix} opt step: {self.optimizer.param_groups[0]["step"]} param name: {name}, {param.norm()}, ', color='lb')

    def transfer_latest_model_states(self, global_decision, prev_state, prev_redundant_state, need_broadcast=True):
        torch.cuda.synchronize()
        st = time.time()
        self.log(f'>>>>>>>> rank: {self.global_rank} start layer transfer')

        prev_num_pipelines, prev_num_stages = self.find_prev_strategy()
        if prev_num_stages == 0:
            return

        old_parts = self.module.get_new_partition(prev_num_stages)

        self.log(f'parts: {old_parts} {(prev_num_pipelines, prev_num_stages)} -> {self.module.parts} {(self.dp_world_size, self.num_stages)}', color='y')

        recv_decisions, redundant_recv_decisions = self.get_recv_decisions(old_parts, self.module.parts, global_decision)
        send_decisions = self.get_send_decisions(recv_decisions)
        redundant_send_decisions = self.get_redundant_send_decisions(redundant_recv_decisions)

        if self.num_pipelines * self.num_stages > prev_num_pipelines * prev_num_stages:
            need_broadcast = need_broadcast and self.dp_world_size > 1
        elif prev_num_stages != self.num_stages or prev_num_pipelines <= self.num_pipelines:
            need_broadcast = need_broadcast and self.dp_world_size > 1
        else:
            need_broadcast = need_broadcast
        need_broadcast = True
        # self.log(f'  broadcast: {need_broadcast}  recv_decisions: {recv_decisions}', color='y')
        # self.log(f'                   send_decisions: {send_decisions}', color='y')
        # self.log(f'                   redundant_recv_decisions: {redundant_recv_decisions}', color='y')
        # self.log(f'rank: {self.global_rank}, may layers: {sorted(list(prev_state.keys()))}', color='y')

        received_state = self.transfer_layers(recv_decisions, send_decisions, prev_state)
        if self.recv_states_from_redundant_module:
            for layer_idx, state in prev_state.items():
                if layer_idx not in received_state:
                    prev_redundant_state[layer_idx] = state
            # for layer_idx in range(self.module.parts[-1]):
            #     if layer_idx not in prev_redundant_state:
            #         prev_redundant_state[layer_idx] = [{}, []]

        received_redundant_state = self.transfer_redundant_layers(redundant_send_decisions, prev_redundant_state)
        # copy to local states
        for layer_idx, state in received_redundant_state.items():
            if layer_idx not in received_state or self.recv_states_from_redundant_module:
                received_state[layer_idx] = state

        # remove ununsed states to free memory
        stage_part_start = self.module.parts[self.stage_id]
        stage_part_end = self.module.parts[self.stage_id + 1]
        stage_part = list(range(stage_part_start, stage_part_end))
        state_keys = list(prev_state.keys())
        for layer_idx in state_keys:
            if layer_idx not in stage_part:
                del prev_state[layer_idx]
        for layer_idx in stage_part:
            if layer_idx in received_state:
                if len(received_state[layer_idx][0]) == 0:
                    del received_state[layer_idx]
            elif layer_idx in prev_state:
                if len(prev_state[layer_idx][0]) == 0:
                    del prev_state[layer_idx]

            if layer_idx not in prev_state or layer_idx not in received_state:
                if self.enable_redundant_model_states or self.redundant_model_only:
                    if layer_idx in prev_redundant_state:
                        received_state[layer_idx] = prev_redundant_state[layer_idx]

        with torch.no_grad():
            self.module.load_layers(received_state, prev_model_state=prev_state)
            self.load_optimizer_state(received_state, prev_state)

        # if need broadcast, do it
        if need_broadcast:
            self.broadcast_latest_model_states()

        del received_state, prev_state, received_redundant_state, prev_redundant_state

        self.show_param_norm(prefix='After MG')

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        ed = time.time()
        self.log(f'>>>>>>>> rank: {self.global_rank} finish layer transfer, takes {ed - st:.3f} seconds')

    def broadcast_latest_model_states(self):
        src_rank = self.grid._topo.get_rank(pipe=self.stage_id, data=0)
        for name, p in self.module.named_parameters():
            # if self.grid.local_rank == 0:
            #     self.log(f'rank: {self.global_rank} broadcast {name}, size: {p.size()}', color='y')
            dist.broadcast(p, src_rank, group=self.mpu.get_data_parallel_group())

        if self.fp16_enabled():
            for _, opt_states_flat in self.optimizer.state.items():
                for key, flat_state in opt_states_flat.items():
                    # if self.grid.local_rank == 0:
                    #     self.log(f'rank: {self.global_rank} broadcast state {key}, {flat_state.size()}', color='y')
                    dist.broadcast(flat_state, src_rank, group=self.mpu.get_data_parallel_group())
            # self.optimizer.refresh_fp32_params()
            # manaully refresh fp32 params
            for i, param_group in enumerate(self.optimizer.fp16_groups):
                flat_fp16_param = self.flatten(param_group)
                self.optimizer.fp32_groups_flat[i].data.copy_(flat_fp16_param.float())
                self.optimizer.fp16_groups_flat[i].data.copy_(flat_fp16_param)
        else:
            if len(self.optimizer.state) == 0:
                # init optimizer state
                for p in self.module.parameters():
                    p.grad = torch.zeros_like(p)
                self.optimizer.step()
                for p in self.module.parameters():
                    del p.grad
                    p.grad = None
            for _, opt_states in self.optimizer.state.items():
                for key, opt_state in opt_states.items():
                    dist.broadcast(opt_state, src_rank, group=self.mpu.get_data_parallel_group())

        # set adam group
        if 'FusedAdam' in self.optimizer.__class__.__name__:
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['step'] = self.global_steps

    def reset_process_groups(self, store, rank=None, world_size=None, master_addr=None, master_port=None, num_pipelines=None, num_stages=None):
        dist.destroy_process_group()

        if rank is None:
            rank = self.global_rank
        if world_size is None:
            world_size = self.world_size
        if store is None:
            assert master_addr is not None
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = str(master_port)
            os.environ['RANK'] = str(rank)
            os.environ['WORLD_SIZE'] = str(world_size)

            dist_store = self.rdzv_handler.setup_comm_store(self.migration_version)
        else:
            dist_store = store
        self.log(f'Reset progress group, store: {dist_store}, backend: {self.dist_backend}, rank: {rank}, world_size: {world_size}, group rank: {self.group_rank}, node: {socket.gethostname()}, master_addr: {master_addr}:{master_port}, intra-version: {self.migration_version}', color='lb')
        dist.init_distributed(self.dist_backend, rank=rank, world_size=world_size, store=dist_store)
        # dist.init_distributed(self.dist_backend, rank=rank, world_size=world_size, store=store)

        if num_pipelines is None:
            num_pipelines = int(os.environ['SPOTDL_NUM_PIPELINES'])
        if num_stages is None:
            num_stages = int(os.environ['SPOTDL_NUM_STAGES'])
        active_group_ranks = self.rdzv_handler.update_rank_status(rank, num_pipelines // self.grid.local_world_size, num_stages)
        return active_group_ranks

    def save_latest_model_states(self, prev_state):
        send_decisions = {}
        parts = self.module.parts
        for stage_id in range(self.num_stages - 1):
            rank = self.grid._topo.get_rank(pipe=stage_id, data=0)
            layers = list(range(parts[stage_id], parts[stage_id + 1]))
            send_decisions[rank] = {'layers': set(layers), 'stage_id': stage_id}
        redundant_node = self.grid._topo.get_rank(pipe=self.num_stages-1, data=0)
        recv_decisions = {redundant_node: send_decisions}
        send_decisions = self.get_send_decisions(recv_decisions)
        self.log(f'   [latest] recv_decisions: {recv_decisions}', color='y')
        self.log(f'   [latest] send_decisions: {send_decisions}', color='y')
        received_state = self.transfer_layers(recv_decisions, send_decisions, prev_state)
        return received_state

    def reconfigure_cluster(self, store, global_decision, recvd_state, failures):
        self.rdzv_handler.write('/rdzv/cluster_status', 'init')
        self.rdzv_handler.write('/rdzv/last_reconfig', self.global_steps)

        del self.pipe_buffers
        # if getattr(self, 'loss_model', None) is not None:
        # if self.loss_model is not None:
        #     del self.loss_model

        # keep original model states
        local_prev_state = self.get_model_state()
        local_prev_state.update(recvd_state)
        # local_prev_redundant_state = self.get_redundant_states(local_prev_state)
        local_prev_redundant_state = {}
        local_prev_redundant_state.update(local_prev_state)

        active_group_ranks = self.reset_process_groups(store, num_pipelines=self.num_pipelines, num_stages=self.num_stages)

        custom_topo = PipeDataParallelTopology(self.num_stages, self.num_pipelines, custom_mapping=global_decision)
        attach_mpu_init_fn = getattr(self.grid._topo, '_attach_mpu_init_fn', None)
        if attach_mpu_init_fn is not None:
            attach_mpu_init_fn(custom_topo)
            custom_topo._attach_mpu_init_fn = attach_mpu_init_fn

        # custom_partitions = self.global_store.get(f'custom-partition-{self.num_stages}')
        custom_partitions = None
        if self.module.class_name == 'GPT2ModelPipe':
            model_fn = self.module.__class__
            model_args = {
                'num_tokentypes': self.module.num_tokentypes,
                'parallel_output': self.module.parallel_output,
                'add_pooler': self.module.add_pooler,
                'topology': custom_topo
            }
        else:
            model_fn = PipelineModule
            model_args = {
                'layers': self.module._layer_specs,
                'num_stages': self.num_stages,
                'topology': custom_topo,
                'loss_fn': self.module.loss_fn,
                'seed_layers': self.module.seed_layers,
                'seed_fn': self.module.seed_fn,
                'base_seed': self.module.base_seed,
                'partition_method': self.module.partition_method,
                'custom_partitions': custom_partitions,
                'activation_checkpoint_interval': self.module.activation_checkpoint_interval,
                'activation_checkpoint_func': self.module.activation_checkpoint_func,
                'checkpointable_layers': self.module.checkpointable_layers,
                'class_name': self.module.class_name
            }
        # delete the old model
        if self.fp16_enabled():
            optimizer = self.optimizer.optimizer
            del self.optimizer.fp32_groups_flat
            del self.optimizer.fp16_groups_flat
            del self.optimizer.fp16_groups
            del self.optimizer.overflow_checker
        else:
            optimizer = self.optimizer

        # if len(local_prev_redundant_state) > 0:
        # if self.is_redundant_node:
        #     del self.redundant_lr_scheduler.optimizer
        #     del self.redundant_lr_scheduler
        #     del self.redundant_optimizer.param_groups
        #     del self.redundant_optimizer.state
        #     del self.redundant_optimizer

        del optimizer.param_groups
        optimizer.param_groups = []
        del optimizer.state
        optimizer.state = collections.defaultdict(dict)

        del self.param_names, self.client_model_parameters
        del self.module.tied_modules, self.module.func_buckets
        if hasattr(self.module, '_attach_get_param_group_fn'):
            _attach_get_param_group_fn = self.module._attach_get_param_group_fn
        else:
            _attach_get_param_group_fn = None
        self._del_client_model()

        model = model_fn(**model_args)
        if _attach_get_param_group_fn is not None:
            model._attach_get_param_group_fn = _attach_get_param_group_fn
            param_groups = model._attach_get_param_group_fn(model)
        else:
            param_groups = [{'params': [p for p in model.parameters() if p.requires_grad]}]
        for param_group in param_groups:
            optimizer.add_param_group(param_group)

        # record batch function
        batch_fn = self.batch_fn

        # Re-init pipeline engine for consistency
        self.__init__(
            args=self.init_args,
            model=model,
            optimizer=optimizer,
            training_data=self.training_data,
            lr_scheduler=self.client_lr_scheduler,
            mpu=model.mpu(),
            failures=failures,
            prev_state=local_prev_state,
            prev_redundant_state=local_prev_redundant_state,
            rdzv_handler=self.rdzv_handler,
            migration_version=0,
        )
        del local_prev_state, local_prev_redundant_state

        self.set_batch_fn(batch_fn)
        self.active_group_ranks = active_group_ranks
        # self.log(f'After reconfigure, batch_fn: {batch_fn}, rank: {self.global_rank}, Active group ranks: {self.active_group_ranks}', color='lb')

    def assign_coordinates(self, fail_ranks, next_dp, next_pp):
        cur_dp, cur_pp = self.dp_world_size, self.num_stages

        # assign coordinates
        global_decision = self.rdzv_handler.get_global_decision()
        global_decision = self.rdzv_handler.convert_global_decision(global_decision, self.dp_world_size)
        current_coordinates = {}
        for info in global_decision:
            if info.rank in fail_ranks or info.rank >= cur_dp * cur_pp:
                continue
            current_coordinates[info.rank] = info.active_coordinates

        old_parts = self.module.parts
        new_parts = self.module.get_new_partition(next_pp)

        required_coordinates = []
        for dp in range(next_dp):
            for pp in range(next_pp):
                cur_coord = (dp, pp)
                required_coordinates.append(cur_coord)

        rank_active_coordinates = {}
        if self.enable_redundant_model_states or self.redundant_model_only:
            # assign redundant coordinate
            for rank, coords in current_coordinates.items():
                if tuple(coords[0]) == (0, cur_pp - 1):
                    redundant_coord = (0, next_pp - 1)
                    rank_active_coordinates[rank] = [redundant_coord]
                    required_coordinates.remove(redundant_coord)

        stage_preference = {}
        for new_pp in range(next_pp):
            prefer = []
            for old_pp in range(cur_pp):
                old_layers = list(range(old_parts[old_pp], old_parts[old_pp + 1]))
                new_layers = list(range(new_parts[new_pp], new_parts[new_pp + 1]))
                prefer.append((old_pp, set(new_layers).difference(set(old_layers))))
            stage_preference[new_pp] = sorted(prefer, key=lambda x: len(x[1]))

        stage_remains = {pp: [] for pp in range(cur_pp)}
        for rank, coords in current_coordinates.items():
            stage_remains[coords[0][1]].append(rank)

        for dp in range(next_dp):
            for pp in range(next_pp):
                cur_coord = (dp, pp)
                if cur_coord not in required_coordinates:
                    continue
                for prev_stage, _ in stage_preference[pp]:
                    if len(stage_remains[prev_stage]) == 0:
                        continue
                    rank = stage_remains[prev_stage].pop(0)
                    if rank not in rank_active_coordinates:
                        rank_active_coordinates[rank] = [cur_coord]
                        required_coordinates.remove(cur_coord)
                    break

        num_participants = self.dp_world_size * self.num_stages #- len(fail_ranks)
        while len(required_coordinates) > 0:
            coordinate = required_coordinates.pop(0)
            for rank in range(num_participants):
                if rank not in rank_active_coordinates and rank not in fail_ranks:
                    rank_active_coordinates[rank] = [coordinate]
                    break

        # Initialize the missing ranks
        for rank in range(num_participants):
            if rank not in rank_active_coordinates and rank not in fail_ranks:
                rank_active_coordinates[rank] = []

        # after assign coordinates, assign new ranks
        rank_map = {}
        for rank, coords in rank_active_coordinates.items():
            if len(coords) == 0:
                continue
            rank_map[rank] = coords[0][0] + coords[0][1] * next_dp

        if self.global_rank not in rank_active_coordinates or len(rank_active_coordinates[self.global_rank]) == 0:
            self.log(f'Rank {self.global_rank} is inactive', color='b')
            self.exit_entrypoint(125)

        # new world info
        global_rank = rank_map[self.global_rank]
        global_world_size = next_dp * next_pp #cur_dp * cur_pp - len(fail_ranks)
        coordinate = rank_active_coordinates[self.global_rank][0]

        if global_rank == 0:
            self.log(f'previous decision: {global_decision}, current_coordinates: {current_coordinates}', color='lb')
            self.log(f'Assigning new coordinates: {rank_active_coordinates}, new_rank map: {rank_map}, world_size: {len(rank_map)}', color='b')

        self.migration_version += 1
        addr_key = f'local_addr_{self.migration_version}'
        port_key = f'local_port_{self.migration_version}'
        if coordinate[0] == 0 and coordinate[1] == 0:
            self.rdzv_handler.set_master_addr_port(self.global_store)
            master_addr, master_port = self.rdzv_handler.get_master_addr_port(self.global_store)
            self.global_store.set(addr_key, master_addr)
            self.global_store.set(port_key, str(master_port))
        else:
            master_addr = self.global_store.get(addr_key).decode(encoding="UTF-8")
            master_port = int(self.global_store.get(port_key))

        active_group_ranks = self.reset_process_groups(None, rank=global_rank, world_size=global_world_size,
                                                       master_addr=master_addr, master_port=master_port,
                                                       num_pipelines=next_dp, num_stages=next_pp)

        # latest global decision
        coordinate_decision = []
        for info in global_decision:
            if info.rank in rank_map:
                new_info = info.__class__(rank_map[info.rank], current_coordinates[info.rank], rank_active_coordinates[info.rank])
                coordinate_decision.append(new_info)

        return active_group_ranks, coordinate_decision, global_rank, global_world_size, coordinate

    def handle_inter_stage_migration(self, fail_ranks, fail_coordinates, next_dp, next_pp, failures, recvd_states={}):
        # handle inter stage migration in a similar way as reparition
        self.handle_repartition(fail_ranks, fail_coordinates, next_dp, next_pp, failures, recvd_states={})

    def handle_repartition(self, fail_ranks, fail_coordinates, next_dp, next_pp, failures, recvd_states={}):
        active_group_ranks, global_decision, global_rank, global_world_size, coordinate = self.assign_coordinates(fail_ranks, next_dp, next_pp)

        custom_topo = PipeDataParallelTopology(next_pp, next_dp, custom_mapping=global_decision)
        attach_mpu_init_fn = getattr(self.grid._topo, '_attach_mpu_init_fn', None)
        if attach_mpu_init_fn is not None:
            attach_mpu_init_fn(custom_topo)
            custom_topo._attach_mpu_init_fn = attach_mpu_init_fn

        del self.pipe_buffers
        # if getattr(self, 'loss_model', None) is not None:
        # if self.loss_model is not None:
        #     del self.loss_model

        # keep original model states
        local_prev_state = self.get_model_state()
        # local_prev_redundant_state = self.get_redundant_states(local_prev_state)
        local_prev_redundant_state = {}
        local_prev_redundant_state.update(recvd_states)
        local_prev_redundant_state.update(local_prev_state)

        # update module
        custom_partitions = None
        if self.module.class_name == 'GPT2ModelPipe':
            model_fn = self.module.__class__
            model_args = {
                'num_tokentypes': self.module.num_tokentypes,
                'parallel_output': self.module.parallel_output,
                'add_pooler': self.module.add_pooler,
                'topology': custom_topo
            }
        else:
            model_fn = PipelineModule
            model_args = {
                'layers': self.module._layer_specs,
                'num_stages': self.num_stages,
                'topology': custom_topo,
                'loss_fn': self.module.loss_fn,
                'seed_layers': self.module.seed_layers,
                'seed_fn': self.module.seed_fn,
                'base_seed': self.module.base_seed,
                'partition_method': self.module.partition_method,
                'custom_partitions': custom_partitions,
                'activation_checkpoint_interval': self.module.activation_checkpoint_interval,
                'activation_checkpoint_func': self.module.activation_checkpoint_func,
                'checkpointable_layers': self.module.checkpointable_layers,
                'class_name': self.module.class_name
            }
        # delete the old model
        if self.fp16_enabled():
            optimizer = self.optimizer.optimizer
            del self.optimizer.fp32_groups_flat
            del self.optimizer.fp16_groups_flat
            del self.optimizer.fp16_groups
            del self.optimizer.overflow_checker
        else:
            optimizer = self.optimizer

        del optimizer.param_groups
        optimizer.param_groups = []
        del optimizer.state
        optimizer.state = collections.defaultdict(dict)

        del self.param_names, self.client_model_parameters
        del self.module.tied_modules, self.module.func_buckets
        if hasattr(self.module, '_attach_get_param_group_fn'):
            _attach_get_param_group_fn = self.module._attach_get_param_group_fn
        else:
            _attach_get_param_group_fn = None
        self._del_client_model()

        model = model_fn(**model_args)
        if _attach_get_param_group_fn is not None:
            model._attach_get_param_group_fn = _attach_get_param_group_fn
            param_groups = model._attach_get_param_group_fn(model)
        else:
            param_groups = [{'params': [p for p in model.parameters() if p.requires_grad]}]
        for param_group in param_groups:
            optimizer.add_param_group(param_group)

        # record batch function
        batch_fn = self.batch_fn

        # Re-init pipeline engine for consistency
        self.__init__(
            args=self.init_args,
            model=model,
            optimizer=optimizer,
            training_data=self.training_data,
            lr_scheduler=self.client_lr_scheduler,
            mpu=model.mpu(),
            prev_state=local_prev_state,
            prev_redundant_state=local_prev_redundant_state,
            failures=failures,
            rdzv_handler=self.rdzv_handler,
            coordinate_decision=global_decision,
            migration_version=self.migration_version,
        )
        del local_prev_state, local_prev_redundant_state

        self.set_batch_fn(batch_fn)
        self.active_group_ranks = active_group_ranks

    def intra_migration_pre_check(self, fail_ranks=None, failures=None):
        if fail_ranks is None:
            fail_ranks, _ = self.convert_failures_to_global_ranks(failures)

        stage_replicas = {stage_id: 0 for stage_id in range(self.num_stages)}
        active_coordinates, waiting_coordinates = {}, {}
        active_ranks = self.grid.get_active_ranks()
        for rank in range(self.grid.world_size):
            if rank in fail_ranks:
                continue
            if rank in active_ranks:
                coord = self.grid._topo.get_coord(rank)
                active_coordinates[rank] = (coord.data, coord.pipe)
                stage_replicas[coord.pipe] += 1
            # else:
            #     coord = self.grid.get_waiting_rank_coordinate(rank)
            #     waiting_coordinates[rank] = (coord.data, coord.pipe)
            #     stage_replicas[waiting_coordinates[rank][1]] += 1

        # decide number of remain pipelines
        remain_num_pipelines = min(list(stage_replicas.values()))
        remain_world_size = self.grid.world_size - len(fail_ranks)

        return remain_num_pipelines, remain_world_size, active_coordinates, waiting_coordinates

    def handle_intra_stage_migration(self, fail_ranks, fail_coordinates, expected_pipelines=0):
        """apply intra-stage migration
        """
        self.log(f'rank: {self.global_rank} find failures {fail_ranks} at step {self.global_steps} and begin to handle...', color='r')
        # fetch latest assignment
        check_ret = self.intra_migration_pre_check(fail_ranks=fail_ranks)
        remain_num_pipelines, remain_world_size, active_coordinates, waiting_coordinates = check_ret
        if remain_num_pipelines < expected_pipelines:
            return 1

        # assign new ranks and coordinates
        rank_map = {}
        ranks_tb_modified, ranks_tb_filled = [], sorted(fail_ranks)
        for rank in range(remain_world_size, self.grid.world_size):
            if rank in ranks_tb_filled:
                ranks_tb_filled.remove(rank)
            else:
                ranks_tb_modified.append(rank)
        assert len(ranks_tb_modified) == len(ranks_tb_filled)

        for i in range(len(ranks_tb_modified)):
            rank_map[ranks_tb_modified[i]] = ranks_tb_filled[i]
        for rank in range(self.grid.world_size):
            if rank not in fail_ranks and rank not in rank_map:
                rank_map[rank] = rank

        # assign new coordinates
        rank_to_coordinate, rank_status = {}, {}
        required_coordinates = []
        for dp_id in range(remain_num_pipelines):
            for stage_id in range(self.num_stages):
                required_coordinates.append((dp_id, stage_id))

        remain_ranks = []
        for rank in rank_map:
            if active_coordinates.get(rank, None) in required_coordinates:
                rank_to_coordinate[rank] = active_coordinates[rank]
                required_coordinates.remove(active_coordinates[rank])
                rank_status[rank] = 'active'
            else:
                remain_ranks.append(rank)

        required_dp_ids = {stage_id: [] for stage_id in range(self.num_stages)}
        for coord in required_coordinates:
            required_dp_ids[coord[1]].append(coord[0])

        remain_waiting_ranks = []
        for rank in sorted(remain_ranks):
            if rank in active_coordinates:
                coord = active_coordinates[rank]
                rank_status[rank] = 'waiting'
                if len(required_dp_ids[coord[1]]) > 0:
                    coord = (min(required_dp_ids[coord[1]]), coord[1])
                    required_dp_ids[coord[1]].remove(coord[0])
                    required_coordinates.remove(coord)
                    rank_status[rank] = 'active'
                rank_to_coordinate[rank] = coord
            else:
                remain_waiting_ranks.append(rank)

        for rank in remain_waiting_ranks:
            coord = (-1, waiting_coordinates[rank][1])
            rank_status[rank] = 'waiting'
            if len(required_dp_ids[coord[1]]) > 0:
                coord = (min(required_dp_ids[coord[1]]), coord[1])
                required_dp_ids[coord[1]].remove(coord[0])
                required_coordinates.remove(coord)
                rank_status[rank] = 'switch_active'
            rank_to_coordinate[rank] = coord
        assert len(required_coordinates) == 0

        # A FIXCODE: early exit if the node is waiting
        new_rank_map = {}
        for rank in rank_map:
            if rank_status[rank] == 'waiting':
                continue
            coords = rank_to_coordinate[rank]
            new_rank_map[rank] = coords[0] + coords[1] * remain_num_pipelines
        rank_map = new_rank_map
        if self.global_rank not in rank_map:
            self.log(f'rank: {self.global_rank} early exit at step {self.global_steps}', color='r')
            self.exit_entrypoint(125)
        remain_world_size = len(rank_map)

        # new world info
        previous_global_rank = self.global_rank
        global_rank = rank_map[self.global_rank]
        global_world_size = remain_world_size
        coordinate = rank_to_coordinate[self.global_rank]
        global_steps = self.global_steps

        self.migration_version += 1
        addr_key = f'local_addr_{self.migration_version}'
        port_key = f'local_port_{self.migration_version}'
        if coordinate[0] == 0 and coordinate[1] == 0:
            self.rdzv_handler.set_master_addr_port(self.global_store)
            master_addr, master_port = self.rdzv_handler.get_master_addr_port(self.global_store)
            self.global_store.set(addr_key, master_addr)
            self.global_store.set(port_key, str(master_port))
        else:
            master_addr = self.global_store.get(addr_key).decode(encoding="UTF-8")
            master_port = int(self.global_store.get(port_key))

        self.reset_process_groups(None, rank=global_rank, world_size=global_world_size,
                                  master_addr=master_addr, master_port=master_port,
                                  num_pipelines=remain_num_pipelines, num_stages=self.num_stages)

        # update pipeline topo and grid
        IntraInfo = collections.namedtuple('IntraInfo', ['rank', 'active_coordinates'])

        intra_decisions = []
        waiting_ranks, switch_active_ranks = [], []
        candidate_pool = {}
        for rank in rank_map:
            if rank_status[rank] in ['active', 'switch_active']:
                info = IntraInfo(rank=rank_map[rank], active_coordinates=[rank_to_coordinate[rank]])
                intra_decisions.append(info)

            if rank_status[rank] == 'switch_active':
                switch_active_ranks.append(rank)
            elif rank_status[rank] == 'waiting':
                waiting_ranks.append(rank)
                candidate_pool[rank_map[rank]] = rank_to_coordinate[rank]
        self.log(f'[Intra-Stage Migration] rank: {global_rank}, world_size: {global_world_size}, coordinate: {coordinate}, '
                 f'intra_decisions: {intra_decisions}, candidate pool: {candidate_pool}, '
                 f'num_pipelines: {remain_num_pipelines}, num_stages: {self.num_stages}, rank_status: {rank_status}', color='b')

        custom_topo = PipeDataParallelTopology(self.num_stages, remain_num_pipelines, custom_mapping=intra_decisions)
        attach_mpu_init_fn = getattr(self.grid._topo, '_attach_mpu_init_fn', None)
        if attach_mpu_init_fn is not None:
            attach_mpu_init_fn(custom_topo)
            custom_topo._attach_mpu_init_fn = attach_mpu_init_fn

        # update comm grid
        self.module.update_comm_grid(topology=custom_topo, candidate_pool=candidate_pool)

        # Re-init deepspeed engine for consistency
        super().__init__(
            args=self.init_args,
            model=self.module,
            optimizer=self.optimizer,
            training_data=self.training_data,
            lr_scheduler=self.client_lr_scheduler,
            mpu=self.module.mpu(),
            rdzv_handler=self.rdzv_handler,
            skip_optimizer_check=True,
        )
        # update optimizer comm
        if self.fp16_enabled():
            self.optimizer.mpu = self.module.mpu()
            self.optimizer.overflow_checker.mpu = self.module.mpu()

        # update pipeline engine for consistency
        self.micro_batch_size = self.train_micro_batch_size_per_gpu()
        self.micro_batches = self.gradient_accumulation_steps()
        self.log(f'[Intra-Stage Migration] step: {global_steps}, global_batch_size: {self.train_batch_size()}, '
                 f'micro_batch_size: {self.micro_batch_size}, micro_batches: {self.micro_batches}', color='b')

        self.grid = self.module._grid
        self.global_rank = self.grid.get_global_rank()
        self.global_steps = global_steps

        assert self.dp_world_size == self.grid.data_parallel_size

        self.is_pipe_parallel = self.grid.pipe_parallel_size > 1
        self.is_data_parallel = self.grid.data_parallel_size > 1
        self.is_model_parallel = self.grid.model_parallel_size > 1

        if self.grid.is_active:
            self._init_p2p_comm()

        # re-configure redundant communication
        self._configure_redundant_comm()
        return 0

    def check_failure_and_exit(self, failures):
        for rank_key, step in failures.items():
            if step == self.global_steps:
                if int(rank_key) == self.group_rank:
                    self.exit_entrypoint(13)

    def convert_failures_to_global_ranks(self, failures):
        failed_group_ranks = []
        for rank_key, step in failures.items():
            if step == self.global_steps:
                # if int(rank_key) == self.group_rank:
                #     self.exit_entrypoint(13)

                if int(rank_key) in self.active_group_ranks:
                    failed_group_ranks.append(int(rank_key))

        fail_ranks, fail_coordinates = [], []
        if len(failed_group_ranks) > 0:
            try:
                _, state = self.rdzv_handler._rdzv_impl.get_rdzv_state()
            except:
                state = None
            if state is None or state['status'] != 'final':
                return [], []

            version = state['version']
            for group_rank in failed_group_ranks:
                fail_coord_key = self.rdzv_handler._rdzv_impl.get_path(
                    f'/rdzv/v_{version}/rank_{group_rank}_coordinates'
                )
                fail_coordinates = json.loads(self.rdzv_handler._rdzv_impl.client.get(fail_coord_key).value)

                if len(fail_coordinates) > 0:
                    fail_coord = fail_coordinates[0]
                    fail_rank = fail_coord[0] + fail_coord[1] * self.dp_world_size
                    fail_coordinates.append(fail_coord)
                    fail_ranks.append(fail_rank)

        return fail_ranks, fail_coordinates

    def check_preemption_and_handle(self, failures):
        fail_ranks, fail_coordinates = self.convert_failures_to_global_ranks(failures)
        # get next planned strategy.
        next_strategy = self.rdzv_handler.get_strategy_plan(self.global_steps)
        if next_strategy is not None:
            node_num_change = next_strategy[0]
            next_dp, next_pp = next_strategy[1]
            need_migration = (next_dp * self.grid.local_world_size != self.dp_world_size) or (next_pp != self.num_stages)
        else:
            need_migration = False

        status = 0
        if len(fail_ranks) > 0 or need_migration:
            recvd_state = {}
            if self.save_latest:
                local_prev_state = self.get_model_state(delete_state=False)
                recvd_state.update(self.save_latest_model_states(local_prev_state))
                del local_prev_state

            self.check_failure_and_exit(failures)
            self.rdzv_handler.update_prev_strategy((self.dp_world_size, self.num_stages))
            torch.cuda.synchronize()
            # decide how to handle this preemptions

            assert next_dp * next_pp * self.grid.local_world_size <= self.dp_world_size * self.num_stages - len(fail_ranks)

            if next_pp != self.num_stages:
                # if the next pipeline depth is not the same as current, we need to handle this preemption.
                status = self.handle_repartition(fail_ranks, fail_coordinates, next_dp, next_pp, failures, recvd_states=recvd_state)
            else:
                # if the next pipeline depth is the same as current,
                # we first try to handle this preemption with intra-stage migration.
                status = self.handle_intra_stage_migration(fail_ranks, fail_coordinates, expected_pipelines=next_dp)
                if status:
                    # if intra-stage migration fails, we need to handle this preemption.
                    status = self.handle_inter_stage_migration(fail_ranks, fail_coordinates, next_dp, next_pp, failures, recvd_states=recvd_state)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return status

    def delete_previous_state(self):
        del self.pipe_buffers
        # if getattr(self, 'loss_model', None) is not None:
        # if self.loss_model is not None:
        #     del self.loss_model

        # delete the old model
        if self.fp16_enabled():
            optimizer = self.optimizer.optimizer
            del self.optimizer.fp32_groups_flat
            del self.optimizer.fp16_groups_flat
            del self.optimizer.fp16_groups
            del self.optimizer.overflow_checker
        else:
            optimizer = self.optimizer

        del optimizer.param_groups
        optimizer.param_groups = []
        del optimizer.state
        optimizer.state = collections.defaultdict(dict)
        del optimizer

        del self.param_names, self.client_model_parameters
        del self.module.tied_modules, self.module.func_buckets
        self._del_client_model()

        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()

    def handle_migration(self, failures):
        recvd_state = {}
        if self.save_latest:
            self.show_param_norm(prefix='Before MGG')
            local_prev_state = self.get_model_state(delete_state=False)
            recvd_state.update(self.save_latest_model_states(local_prev_state))
            del local_prev_state

        if failures.get(str(self.group_rank), -1) == self.global_steps:
            self.log(f'We are doing migration and I will die soon anyway. I\'m leaving.', color='r')
            self.exit_entrypoint(13)

        self.log(f'Triggered a migration on global step {self.global_steps}', color='lb')
        if self.grid.local_rank == 0:
            store, group_rank, group_world_size, num_pipelines, num_stages, global_decision = self.rdzv_handler.next_rendezvous(self.group_rank)
            # assign woker group info, setup env info
            Spec = collections.namedtuple('WorkerSpec', ['role', 'local_world_size'])
            spec = Spec('default', 1)
            self.rdzv_handler.assign_worker_ranks(store, group_rank, group_world_size, spec, num_pipelines, num_stages, global_decision)

        if self.grid.local_world_size > 1:
            if self.grid.local_rank > 0:
                # wait for rank 0 to finish the group assignment
                time.sleep(5.5)
            # broadcast the group info
            src_rank = (self.global_rank // self.grid.local_world_size) * self.grid.local_world_size
            if self.grid.local_rank == 0:
                infos = torch.tensor([group_rank, group_world_size, num_pipelines, num_stages], dtype=torch.int32).cuda()
            else:
                infos = torch.tensor([0, 0, 0, 0], dtype=torch.int32).cuda()
            dist.broadcast(infos, src_rank, group=self.grid.local_proc_group)
            group_rank, group_world_size, num_pipelines, num_stages = infos.cpu().numpy().tolist()
            num_pipelines = num_pipelines * self.grid.local_world_size
            store = self.rdzv_handler.setup_kv_store()
            global_decision = self.rdzv_handler.get_global_decision()

        os.environ['GROUP_RANK'] = str(group_rank)
        os.environ['GROUP_WORLD_SIZE'] = str(group_world_size)
        self.group_rank = group_rank

        # reassign the global rank
        global_decision, rank, world_size, master_addr, master_port = self.rdzv_handler.reassign_global_rank(global_decision, store)
        if rank < 0:
            self.log(f'Node {socket.gethostname()} is not active (world_size {world_size})', color='r')
            self.exit_entrypoint(125)

        self.log(f'Rendezvous complete! rank: {rank}, world_size: {world_size}, num_pipelines: {num_pipelines}, num_stages: {num_stages}, master addr: {master_addr}:{master_port}', color='lb')
        # if rank == 0:
        #     self.log(f'Global decision: {global_decision}', color='lb')
        self.global_rank = rank
        self.world_size = world_size
        self.global_store = store
        self.num_pipelines = num_pipelines
        self.num_stages = num_stages

        self.reconfigure_cluster(store, global_decision, recvd_state, failures)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def train_batch(self, data_iter=None, get_dataloader_fn=None):
        """Progress the pipeline to train the next batch of data. The engine will ingest
        ``self.train_batch_size()`` total samples collectively across all workers.


        An iterator that over training data should be provided as an argument
        unless ``deepspeed.initialize()`` was provided a training set. In that event,
        the training data will automatically be read.


        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator, optional): Iterator of training data.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        self.rdzv_handler.write('/rdzv/cluster_status', 'train')
        if self.is_log_node:
            self.log(f'STARTING BATCH {self.global_steps} with coordinates {self.coordinates}')

        # update global_step for nodes that are not involved in training
        if self.global_rank not in self.grid.get_active_ranks():
            time.sleep(50 / 1000) # 100ms
            global_step = int(self.global_store.get('global-steps'))
            self.global_steps = max(global_step, self.global_steps)

        # global should_stop
        # if should_stop:
        #     self.fail_lock.acquire()
        #     failures = json.loads(self.global_store.get('failures'))
        #     already_deleted = []
        #     for rank, step in failures.items():
        #         if step < self.global_steps:
        #             already_deleted.append(rank)

        #     for rank in already_deleted:
        #         del failures[rank]

        #     failures[self.global_rank] = self.global_steps + 1
        #     self.global_store.set('failures', json.dumps(failures))
        #     should_stop = False
        #     self.log(f'------- EXITING, set failures to {failures}', color='r')
        #     self.fail_lock.release()

        # self.report_memory(force=True)
        failures = json.loads(self.global_store.get('failures'))
        real_failures = {}
        for key, step in failures.items():
            if step >= self.global_steps:
                real_failures[key] = step
        if len(real_failures) > 0:
            if self.global_rank in self.grid.get_active_ranks():
                self.log(f'FAILURES {failures} at step {self.global_steps}', color='r')
            pre_check_res = self.intra_migration_pre_check(failures=failures)
            remain_num_pipelines = pre_check_res[0]
            remain_world_size = pre_check_res[1]
        else:
            remain_num_pipelines = self.grid.get_data_parallel_world_size()
            remain_world_size = self.dp_world_size * self.num_stages
        node_status = (remain_num_pipelines, self.num_stages, remain_world_size)

        # check whether do we need to do reconfiguration, which takes some time
        if not self.join and self.rdzv_handler.should_migration(self.global_steps, failures, node_status):
            self.rdzv_handler.update_prev_strategy((self.dp_world_size // self.grid.local_world_size, self.num_stages))
            self.handle_migration(failures)
            failures = {}

            if get_dataloader_fn is not None:
                return (self.global_steps, self.epoch)
        else:
            # check failures again because we may need to handle preemptions in a lightweight way
            # e.g. runtime intra-stage and inter-stage migration, repartition
            self.check_preemption_and_handle(failures)

        if self.join:
            self.join = False

        # For node in current process grop but not used, we let them do nothing.
        # they will join the training when they are needed via migration.
        # these nodes will check current training step state to see if they may be needed.
        if self.global_rank not in self.grid.get_active_ranks():
            return 0.0

        self.timers('global_step').start()

        if not torch._C.is_grad_enabled():
            raise RuntimeError(
                f'train_batch() requires gradients enabled. Use eval_batch() instead.')

        # Curriculum learning could change activation shape
        if self.curriculum_enabled():
            new_difficulty = self.curriculum_scheduler.update_difficulty( \
                self.global_steps + 1)
            if self.global_steps == 0 or self.curriculum_scheduler.first_step:
                self.reset_activation_shape()
                self.curriculum_scheduler.first_step = False
            elif new_difficulty != self.curriculum_scheduler.get_difficulty( \
                self.global_steps):
                self.reset_activation_shape()

        self.get_dataloader_fn = get_dataloader_fn
        if data_iter:
            self.set_dataiterator(data_iter)

        self.module.train()
        self.total_loss = None
        self._compute_loss = True

        # Do the work
        self.timers('train_batch').start()

        # self.report_memory(force=True)

        # First trail
        sched = schedule.TrainSchedule(micro_batches=self.micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        schedule_status = self._exec_schedule(sched)
        # FIXME: temporary disable data manager
        # ### add mappings from sample IDs to its microbatch
        # self.minibatch_sample_counter = 0
        # self.minibatch_effective_samples = 0
        # self.minibatch_sample_ids = set()
        # self.microbatch_ptr = 0
        # self.microbatch_counter = 0
        # self.microbatch_to_sampleids = {}
        # ###
        # self._exec_schedule(sched)
        # ### if there is preemption, self.minibatch_sample_ids is not empty! update it to
        # ### TODO: when detecting a preemption, need to sync self.minibatch_sample_ids with self.lost_samples_epoch
        # ###       also, need to write self.lost_samples_epoch into etcd so that other complete pipelines can read
        # ### TODO: consider writing remaining sample IDs of this minibatch, this epoch?
        # self.lost_samples_epoch = self.lost_samples_epoch.union(self.minibatch_sample_ids)
        # ### add number of effective samples in this minibatch to pipeline counter
        # self.num_effective_samples_epoch += self.minibatch_effective_samples
        # ### reset counter and cache for data manager
        # self.minibatch_sample_counter = 0
        # self.minibatch_sample_ids = None
        # self.minibatch_effective_samples = 0
        # self.microbatch_ptr = 0
        # self.microbatch_counter = 0
        # self.microbatch_to_sampleids = None
        # ###
        self.agg_train_loss = self._aggregate_total_loss()
        self.epoch_step += 1

        self.timers('train_batch').stop()

        if self.is_log_node:
            self.global_store.set('global-steps', str(self.global_steps))

        if self.global_steps % self.steps_per_print() == 0:
            if self.is_log_node:
                elapsed = self.timers('train_batch').elapsed(reset=True) / 1000.0
                iter_time = elapsed / self.steps_per_print()
                tput = self.train_batch_size() / iter_time
                print(f'steps: {self.global_steps} '
                      f'loss: {self.agg_train_loss:0.4f} '
                      f'iter time (s): {iter_time:0.3f} '
                      f'samples/sec: {tput:0.3f}')

        # Monitoring
        if self.is_log_node and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/train_loss',
                                    self.agg_train_loss.mean().item(),
                                    self.global_samples)]
            self.monitor.write_events(self.summary_events)

        if self.wall_clock_breakdown(
        ) and self.global_steps % self.steps_per_print() == 0:
            self.timers.log([
                'pipe_send_output',
                'pipe_send_grad',
                'pipe_recv_input',
                'pipe_recv_grad'
            ])

        self.timers('global_step').stop()
        elapsed = self.timers('global_step').elapsed(reset=True)
        self.report_memory()
        if self.is_log_node:
            self.log(f'FINISHED BATCH {self.global_steps - 1} with coordinates {self.coordinates} in {elapsed} ms ({self.dp_world_size}x{self.num_stages})')
        return self.agg_train_loss

    def eval_batch(self,
                   data_iter,
                   return_logits=False,
                   compute_loss=True,
                   num_microbatches=None,
                   reduce_output='avg'):
        """Evaluate the pipeline on a batch of data from ``data_iter``. The
        engine will evaluate ``self.train_batch_size()`` total samples
        collectively across all workers.

        This method is equivalent to:

        .. code-block:: python

            module.eval()
            with torch.no_grad():
                output = module(batch)

        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator): Iterator of data to evaluate.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        self.eval_return_logits = return_logits
        self.module.eval()

        self.eval_correct = 0
        self.eval_total = 0

        # Curriculum learning could change activation shape
        if self.curriculum_enabled():
            new_difficulty = self.curriculum_scheduler.update_difficulty( \
                self.global_steps + 1)
            if self.global_steps == 0 or self.curriculum_scheduler.first_step:
                self.reset_activation_shape()
                self.curriculum_scheduler.first_step = False
            elif new_difficulty != self.curriculum_scheduler.get_difficulty( \
                self.global_steps):
                self.reset_activation_shape()

        eval_output = None

        self._compute_loss = compute_loss

        # Use the provided data iterator
        train_iterator = self.data_iterator
        self.set_dataiterator(data_iter)

        # Do the work
        sched = schedule.InferenceSchedule(micro_batches=self.micro_batches if num_microbatches is None else num_microbatches,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)

        # prevent dead-lock with multiple evals sequence
        dist.barrier()

        with torch.no_grad():
            self._exec_schedule(sched)

        if self.is_last_stage():
            eval_output = self._reduce_outputs(self.fwd_outputs, reduce=reduce_output)

        if compute_loss:
            eval_output = self._bcast_pipe_scalar(eval_output)

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/eval_loss',
                                    eval_output.mean().item(),
                                    self.global_samples)]
            self.monitor.write_events(self.summary_events)

        # Restore the training iterator
        self.set_dataiterator(train_iterator)

        # Reset any buffers that may have been populated during the forward passes.
        #ds_checkpointing.reset()
        self.eval_return_logits = False
        if return_logits:
            outputs = self.outputs
            self.outputs = None
            if self.eval_total > 0:
                acc = self.eval_correct / self.eval_total
                self.log(f'Eval accuracy: {acc * 100:.2f}% ({self.eval_correct} / {self.eval_total}', color='y')
            else:
                acc = 0
            return eval_output, outputs, acc
        return eval_output

    def set_train_batch_size(self, train_batch_size):
        """Adjust the global batch size by increasing or decreasing the number of
        micro-batches (i.e., gradient accumulation steps). The size of each micro-batch
        (i.e., ``train_micro_batch_size_per_gpu``) is not changed.
        Args:
            train_batch_size (int): The new global batch size for training.
        Raises:
            ValueError: if ``train_batch_size`` is not divisible by the
                configured micro-batch size and data parallelism.
        """
        super().set_train_batch_size(train_batch_size)
        self.micro_batches = self.gradient_accumulation_steps()

    def is_first_stage(self):
        """True if this process is in the first stage in the pipeline."""
        return self.stage_id == 0

    def is_last_stage(self):
        """True if this process is in the last stage in the pipeline."""
        return self.stage_id == self.num_stages - 1

    def _reduce_outputs(self, outputs, reduce='avg', reduce_dp=True):
        if reduce is None:
            return outputs

        if reduce.lower() == 'avg':
            # first sum over all microbatches
            if torch.is_tensor(outputs[0]):
                reduced = sum(outputs)
            else:
                assert isinstance(outputs, (list, tuple))
                reduced = [torch.zeros_like(o) for o in outputs[0]]
                for idx, out in outputs:
                    reduced[idx] += out

            # Average over the microbatches
            reduced = self._scale_loss_by_gas(reduced)

            # Average over DP groups
            if reduce_dp and self.is_data_parallel:
                if torch.is_tensor(reduced):
                    dist.all_reduce(reduced, group=self.mpu.get_data_parallel_group())
                    reduced /= self.dp_world_size
                else:
                    for idx in range(len(reduced)):
                        dist.all_reduce(reduced[idx],
                                        group=self.mpu.get_data_parallel_group())
                        reduced[idx] /= self.dp_world_size

            return reduced
        else:
            raise NotImplementedError(f'reduction type {reduce} not supported.')

    def _bcast_pipe_scalar(self, data, src_rank=None, dtype=torch.float32):
        # Default to last stage (e.g., for broadcasting loss)
        if src_rank is None:
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
        assert src_rank in self.grid.pp_group

        if self.global_rank == src_rank:
            result = data.clone().detach()
        else:
            result = torch.Tensor([0.]).type(dtype).to(self.device)

        dist.broadcast(tensor=result,
                       src=src_rank,
                       group=self.mpu.get_pipe_parallel_group())

        return result

    def _aggregate_total_loss(self):
        # Scale loss, average among DP ranks, and bcast loss to the rest of my DP group
        if self.is_last_stage():
            loss = self._scale_loss_by_gas(self.total_loss)
            self.dp_group_loss = loss.clone().detach()

            ## Average loss across all data-parallel groups
            agg_loss = self.dp_group_loss.clone().detach()
            #print(f'RANK={self.global_rank} bcast SENDER src={self.global_rank} group={self.grid.pp_group}', flush=True)

            # NOTE: Temporarily disable for development
            # if self.is_data_parallel:
            #     dist.all_reduce(agg_loss, group=self.mpu.get_data_parallel_group())
            #     agg_loss /= self.dp_world_size

            # assert self.global_rank in self.grid.pp_group
            # losses = torch.Tensor([self.dp_group_loss, agg_loss]).to(self.device)
            # dist.broadcast(tensor=losses,
            #                src=self.global_rank,
            #                group=self.mpu.get_pipe_parallel_group())

        else:
            # NOTE: Temporarily disable for development
            # # Get loss from last stage
            # src_rank = self.grid.stage_to_global(self.num_stages - 1)
            # assert src_rank in self.grid.pp_group
            # losses = torch.Tensor([0., 0.]).to(self.device)
            # dist.broadcast(tensor=losses,
            #                src=src_rank,
            #                group=self.grid.get_pipe_parallel_group())
            # self.dp_group_loss = losses[0].clone().detach()
            # agg_loss = losses[1].clone().detach()
            agg_loss = torch.tensor(0)

        return agg_loss

    def set_dataloader(self, loader):
        """"""
        if self.is_first_stage() or self.is_last_stage():
            self.training_dataloader = loader
            self.data_iterator = iter(self.training_dataloader)

    def set_dataiterator(self, iterator):
        """ Store an iterator to sample for training data. """
        if self.is_first_stage() or self.is_last_stage():
            self.training_dataloader = None
            self.data_iterator = iterator

    def set_batch_fn(self, fn):
        """Execute a post-processing function on input data.

        Args:
            fn (function): The function to run.
        """
        self.batch_fn = fn

    def is_gradient_accumulation_boundary(self):
        """True if the engine is executing a gradient reduction or optimizer step instruction.

        This is overridden from :class:`DeepSpeedEngine` to force reductions
        and steps when the pipeline engine is instructed to do so.

        Returns:
            bool: whether reductions and optimizer steps should occur.
        """
        return self._force_grad_boundary

    def log_for_device(self, *msg):
        if LOG_STAGE == self.stage_id or LOG_STAGE == -1:
            if DATA_PARALLEL_ID == self.grid.data_parallel_id or DATA_PARALLEL_ID == -1:
                print(
                    f'RANK={dist.get_rank()} '
                    f'PIPE-ID={self.stage_id} '
                    f'DATA-ID={self.grid.data_parallel_id} '
                    f'MBATCH-ID={self.microbatch_id} '
                    f'STEP-ID={self.log_batch_step_id} '
                    '::',
                    *msg,
                    flush=True)

    def tput_log(self, *msg):
        if self.global_rank == 0 and self.global_steps % self.steps_per_print() == 0:
            print(*msg)

    def _next_batch(self):
        # If using 3D parallelism, only some first-stage ranks may do IO
        batch = None
        if self.data_iterator is not None:
            if os.environ.get('SPOTDL_SYNETHIC', None):
                # batch = (torch.randn(8, 3, 224, 224).cuda(), torch.zeros(8).long().cuda())
                batch = (torch.randn(4, 1024, 1024).cuda(), torch.randn(4, 1024).cuda())
            else:
                try:
                    batch = next(self.data_iterator)
                except StopIteration:
                    if self.get_dataloader_fn is not None:
                        self.epoch += 1
                        self.epoch_step = 0
                        self.data_iterator = self.get_dataloader_fn(self.grid.data_parallel_size, self.global_rank, self.epoch, self.epoch_step)
                        batch = next(self.data_iterator)
                    else:
                        self.data_iterator = iter(self.training_dataloader)
                        batch = next(self.data_iterator)

            # FIXME: temporary disable data manager
            # # check if having reached epoch end repeatively. If so, load dropped_dataset
            # if self.data_iterator.reach_epoch_end():
            #     # TODO: write number of effective samples to etcd here

            #     self.num_effective_samples_epoch = 0 # reset effective sample before next epoch begins
            #     replay_data_iterator = self.load_replay_data_iterator()
            #     self.data_iterator.update_replay_iterator(replay_data_iterator)

        # Any post-processing, like broadcasting across a slice-parallel group.
        if self.batch_fn:
            batch = self.batch_fn(batch)

        if self.fp16_enabled():
            if torch.is_tensor(batch[0]):
                if batch[0].is_floating_point():
                    batch = (batch[0].half(), batch[1])
        return batch

    def load_replay_data_iterator(self):
        replay_dataloader = []
        if self.lost_samples_epoch:
            replay_dataset = [self.dataset[idx] for idx in self.lost_samples_epoch]
            self.lost_samples_epoch = None # reset sample IDs occurred this epoch
            self.lost_samples_epoch = set() # TODO: will change this logic, where lost_samples_epoch should read from etcd all dropped sample IDs, and then average IDs across all remaining pipelines
            replay_sampler = torch.utils.data.distributed.DistributedSampler( # has this step already averaged samples across remaining pipelines?
                replay_dataset,
                num_replicas=self.dp_world_size,
                rank=self.mpu.get_data_parallel_rank(),
                shuffle=False)
            # Build a loader.
            replay_dataloader = self.deepspeed_io(replay_dataset,
                                                  data_sampler=replay_sampler,
                                                  build_replay_loader=True)
        return iter(replay_dataloader)

    def _exec_forward_pass(self, buffer_id):
        self.tput_timer.start()
        self.mem_status('BEFORE FWD', reset_max=True)

        if isinstance(self.pipe_buffers['inputs'][buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][buffer_id])
        else:
            inputs = self.pipe_buffers['inputs'][buffer_id].clone()

        # collect the partitioned input from the previous stage
        if self.is_pipe_partitioned and not self.is_first_stage():
            part_input = PartitionedTensor.from_meta(
                meta=inputs[0],
                local_part=inputs[1],
                group=self.grid.get_slice_parallel_group())

            inputs = (part_input.full(), *inputs[2:])
            inputs[0].requires_grad = True
            # skip mask
            #inputs[1].requires_grad = True
            part_input = None
            inputs = inputs[0] if len(inputs) == 1 else inputs
            self.pipe_buffers['inputs'][buffer_id] = inputs

        # Zero out the gradients each time we use the tensor because only the data in
        # tensor changes across batches
        self._zero_grads(inputs)

        outputs = super().forward(inputs, stage_id=self.stage_id)
        # issue redundant gradients receiver after forward to overlap
        # computation with redundant gradient communication
        if self.enable_redundant_model_states and self.is_redundant_node:
            if self.is_redundant_gradients_ready:
                num_receivers = self.redundant_gradients_recv_num['steady_forward_phase'].pop(0)
                self.recv_redundant_gradients(num_receivers)

        # Partition the outputs if we are not the last stage
        if self.is_pipe_partitioned and not self.is_last_stage():
            if isinstance(outputs, tuple):
                first_output = outputs[0]
                # TODO: Improve pipe partitioning to pass multiple tensors that require grads
                assert all([
                    torch.is_tensor(elt) and elt.requires_grad is False
                    for elt in outputs[1:]
                ])
                outputs_tail = outputs[1:]
            elif torch.is_tensor(outputs):
                first_output = outputs
                outputs_tail = []
            else:
                raise ValueError("expecting a tensor or a tuple of tensors")
            part = PartitionedTensor(tensor=first_output,
                                     group=self.grid.get_slice_parallel_group())
            # Clear the large output data, but save the computation graph
            first_output.data = torch.zeros(1)
            self.pipe_buffers['output_tensors'][buffer_id] = first_output
            # Inject the partitioned tensor into the output before sending
            outputs = (part.to_meta(), part.data(), *outputs_tail)
            part = None

        self.pipe_buffers['outputs'][buffer_id] = outputs

        # Optionally compute loss on the last device
        if self.is_last_stage():
            if self._compute_loss and self.module.loss_fn is not None:
                labels = self.pipe_buffers['labels'][buffer_id]
                self.loss = self.module.loss_fn(outputs, labels)
            else:
                # Some models just return loss from forward()
                self.loss = outputs
            if self.eval_return_logits:
                self.outputs = outputs

                _, predicted = torch.max(outputs.data, 1)
                self.eval_correct += (predicted == labels).sum().item()
                self.eval_total += labels.size(0)
            if isinstance(self.loss, torch.Tensor):
                self.fwd_outputs.append(self.loss.detach())

                if self.total_loss is None:
                    self.total_loss = torch.zeros_like(self.loss)
                self.total_loss += self.loss.detach()
            else:
                self.fwd_outputs.append([l.detach() for l in self.loss])

                if self.total_loss is None:
                    self.total_loss = [torch.zeros_like(l) for l in self.loss]
                for idx, l in enumerate(self.loss):
                    self.total_loss[idx] += l.detach()

    def _exec_backward_pass(self, buffer_id):
        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        self.mem_status('BEFORE BWD', reset_max=True)

        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        if self.is_last_stage():
            super().backward(self.loss)

            # issue redundant gradients receiver after backward to overlap
            # computation with redundant gradient communication
            if self.enable_redundant_model_states and self.is_redundant_node:
                if self.is_redundant_gradients_ready:
                    num_receivers = self.redundant_gradients_recv_num['steady_phase'].pop(0)
                    self.recv_redundant_gradients(num_receivers)

            self.mem_status('AFTER BWD')
            # FIXME: temporary disable data manager
            # if self.is_first_stage(): # if pipeline depth is 1
            #     self.minibatch_effective_samples += len(self.pipe_buffers['outputs'][buffer_id])
            #     completed_sample_ids = self.microbatch_to_sampleids[self.microbatch_id - 1]
            #     del self.microbatch_to_sampleids[self.microbatch_ptr - 1]
            #     self.microbatch_ptr -= 1 # pop microbatch id reversely, given 1F1B schedule
            #     self.minibatch_sample_ids = self.minibatch_sample_ids.difference(completed_sample_ids)
            return

        outputs = self.pipe_buffers['outputs'][buffer_id]

        if self.wall_clock_breakdown():
            self.timers('backward_microstep').start()
            self.timers('backward').start()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner').start()

        # Reconstruct if we previously partitioned the output. We must be
        # careful to also restore the computational graph of the tensors we partitioned.
        if self.is_pipe_partitioned:
            if self.is_grad_partitioned:
                part_output = PartitionedTensor.from_meta(
                    meta=outputs[0],
                    local_part=outputs[1],
                    group=self.grid.get_slice_parallel_group())
                self.pipe_buffers['output_tensors'][buffer_id].data = part_output.full()
                outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[2:])
            else:
                # Already restored from partition
                self.pipe_buffers['output_tensors'][buffer_id].data = outputs[0]
                outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[1:])

        grad_tensors = self.grad_layer
        if self.is_grad_partitioned:
            part_grad = PartitionedTensor.from_meta(
                meta=self.grad_layer[0],
                local_part=self.grad_layer[1],
                group=self.grid.get_slice_parallel_group())
            grad_tensors = (part_grad.full(), *grad_tensors[2:])
            part_grad = None

        if self.bfloat16_enabled() and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.clear_lp_grads()

        # This handles either a single tensor or tuple of tensors.
        if isinstance(outputs, tuple):
            out_tensors = [t for t in outputs if t.is_floating_point()]
            assert len(out_tensors) == len(grad_tensors)
            torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
        else:
            torch.autograd.backward(tensors=(outputs, ), grad_tensors=(grad_tensors, ))

        if self.bfloat16_enabled() and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.update_hp_grads(clear_lp_grads=False)

        # Free up the memory from the output of forward()
        self.pipe_buffers['output_tensors'][buffer_id] = None
        self.pipe_buffers['outputs'][buffer_id] = None
        grad_tensors = None

        # FIXME: temporary disable data manager
        # # remove sample ids of computed microbatches
        # if self.is_first_stage():
        #     self.minibatch_effective_samples += len(outputs)
        #     completed_sample_ids = self.microbatch_to_sampleids[self.microbatch_ptr - 1]
        #     del self.microbatch_to_sampleids[self.microbatch_ptr - 1]
        #     self.microbatch_ptr -= 1 # pop microbatch id reversely, given 1F1B schedule
        #     self.minibatch_sample_ids = self.minibatch_sample_ids.difference(completed_sample_ids)
        # ######

        if self.wall_clock_breakdown():
            self.timers('backward_inner').stop()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward').stop()
            self.timers('backward_microstep').stop()

        self.mem_status('AFTER BWD')

    def _exec_load_micro_batch(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('batch_input').start()

        batch = self._next_batch()

        if self.is_first_stage():
            loaded = None
            # FIXME: temporary disable data manager
            # ### sample_ids is included thanks to TraceDataset in DeepSpeedDataLoader
            # sample_ids = set(batch[2].tolist()) if isinstance(batch[2], torch.Tensor) else set(batch[2])
            # self.minibatch_sample_ids = self.minibatch_sample_ids.union(sample_ids)
            # self.microbatch_to_sampleids[self.microbatch_ptr] = sample_ids
            # self.microbatch_ptr += 1
            # self.microbatch_counter += 1 # check if number of microbatches is 32
            # ###
            if torch.is_tensor(batch[0]):
                loaded = batch[0].clone().to(self.device).detach()
                loaded.requires_grad = loaded.is_floating_point()
            else:
                assert isinstance(batch[0], tuple)
                # Assume list or tuple
                loaded = []
                for x in batch[0]:
                    assert torch.is_tensor(x)
                    mine = x.clone().detach().to(self.device)
                    mine.requires_grad = mine.is_floating_point()
                    loaded.append(mine)
                loaded = tuple(loaded)

            self.pipe_buffers['inputs'][buffer_id] = loaded

        if self.is_last_stage():
            loaded = batch[1]
            if torch.is_tensor(batch[1]):
                loaded = batch[1].to(self.device)
            elif isinstance(batch[1], tuple):
                loaded = []
                for x in batch[1]:
                    assert torch.is_tensor(x)
                    x = x.to(self.device).detach()
                    loaded.append(x)
                loaded = tuple(loaded)

            self.pipe_buffers['labels'][buffer_id] = loaded

        if self.wall_clock_breakdown():
            self.timers('batch_input').stop()

    def _send_tensor_meta(self, buffer, recv_stage):
        """ Communicate metadata about upcoming p2p transfers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape
        """
        send_bytes = 0
        if isinstance(buffer, torch.Tensor):
            type_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            send_shape = torch.LongTensor(data=buffer.size()).to(self.device)
            send_ndims = torch.LongTensor(data=[len(buffer.size())]).to(self.device)
            p2p.send(send_ndims, recv_stage)
            p2p.send(send_shape, recv_stage)
            send_bytes += _tensor_bytes(buffer)
        elif isinstance(buffer, list):
            assert (False)
            type_tensor = torch.LongTensor(data=[1]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            for tensor in buffer:
                assert isinstance(tensor, torch.Tensor)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
                send_bytes += _tensor_bytes(tensor)
        elif isinstance(buffer, tuple):
            type_tensor = torch.LongTensor(data=[2]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            for idx, tensor in enumerate(buffer):
                assert isinstance(tensor, torch.Tensor)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                send_dtype = torch.LongTensor(data=[self.DTYPE_TO_ID[tensor.dtype]]).to(
                    self.device)
                p2p.send(send_dtype, recv_stage)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
                # Useful for performance debugging.
                '''
                new_bytes = _tensor_bytes(tensor)
                send_bytes += _tensor_bytes(tensor)
                # Useful for performance debugging.
                if self.grid.data_parallel_id == 0:
                    print(
                        f'STAGE={self.stage_id} pipe-send-volume[{idx}]: shape={send_shape} {new_bytes/1024**2:0.2f}MB'
                    )
                '''
        else:
            raise NotImplementedError(f'Could not send meta type {type(buffer)}')

        # Useful for performance debugging.
        '''
        if self.grid.data_parallel_id == 0:
            print(f'STAGE={self.stage_id} pipe-send-volume: {send_bytes/1024**2:0.2f}MB')
        '''

    def _recv_tensor_meta(self, send_stage):
        """Receive metadata about upcoming p2p transfers and return allocated buffers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape

        Returns:
            Allocated buffer for receiving from send_stage.
        """

        type_tensor = torch.LongTensor(data=[0]).to(self.device)
        p2p.recv(type_tensor, send_stage)
        recv_type = type_tensor.item()

        # A single tensor will be sent.
        if recv_type == 0:
            recv_ndims = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(recv_ndims, send_stage)
            recv_ndims = recv_ndims.item()
            recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
            p2p.recv(recv_shape, send_stage)
            recv_shape = recv_shape.tolist()
            return self._allocate_buffer(recv_shape, num_buffers=1)[0]

        # List or tuple of tensors
        elif recv_type == 1 or recv_type == 2:
            count_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(count_tensor, send_stage)
            num_tensors = count_tensor.item()
            recv_shapes_and_dtypes = []
            for idx in range(num_tensors):
                recv_dtype = torch.LongTensor(data=[0]).to(self.device)
                p2p.recv(recv_dtype, send_stage)
                recv_dtype = self.ID_TO_DTYPE[recv_dtype.item()]
                recv_ndims = torch.LongTensor(data=[0]).to(self.device)
                p2p.recv(recv_ndims, send_stage)
                recv_ndims = recv_ndims.item()
                recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
                p2p.recv(recv_shape, send_stage)
                recv_shapes_and_dtypes.append((recv_shape.tolist(), recv_dtype))

            buffers = self._allocate_buffers(recv_shapes_and_dtypes, num_buffers=1)[0]
            # Convert to tuples if requested.
            if recv_type == 2:
                buffers = tuple(buffers)
            return buffers

        else:
            raise NotImplementedError(f'Could not receive type {type(recv_type)}')

    def _exec_send_activations(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]

        # NCCL does not like to send torch.BoolTensor types, so cast the mask to half().
        # We could do char, but with half() we can eventually flatten with other fp16
        # messages (TODO)
        if self.has_attention_mask or self.has_bool_tensors:
            outputs = list(outputs)
            outputs[-1] = outputs[-1].half()
            outputs = tuple(outputs)

        if self.first_output_send:
            self.first_output_send = False
            self._send_tensor_meta(outputs, self.next_stage)

        def send_handler(stage):
            if isinstance(outputs, torch.Tensor):
                p2p.send(outputs, stage)
            elif isinstance(outputs, tuple):
                for idx, buffer in enumerate(outputs):
                    p2p.send(buffer, stage)
            else:
                raise NotImplementedError('Could not send output of type '
                                          f'{type(outputs)}')

        try:
            send_handler(self.next_stage)
        except Exception as e:
            raise
            dp_id = self.grid.get_data_parallel_id()
            failed_rank = self.grid._topo.get_rank(data=dp_id, pipe=self.next_stage)
            self.global_store.set(f'exception-{failed_rank}', '1')
            self.log(f'---- STEP {self.global_steps}, SEND ACTS TO {failed_rank} '
                     f'(pipe={self.next_stage}, data={dp_id}) FAILED!!!! RESORTING TO FALLBACK {e}', color='r')
            raise NextStageException(e, dp_id=dp_id, stage_id=self.next_stage)

        # Restore the boolean tensor
        if self.has_attention_mask or self.has_bool_tensors:
            outputs = list(outputs)
            outputs[-1] = outputs[-1].bool()
            outputs = tuple(outputs)

        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').stop()

    def _exec_send_grads(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').start()

        inputs = self.pipe_buffers['inputs'][buffer_id]

        # Partition the gradient
        if self.is_grad_partitioned:
            if isinstance(inputs, tuple):
                first_input = inputs[0]
                assert all([torch.is_tensor(elt) for elt in inputs[1:]])
                inputs_grad_tail = [
                    elt.grad for elt in inputs[1:] if elt.grad is not None
                ]
            elif torch.is_tensor(inputs):
                first_input = inputs
                inputs_grad_tail = []
            else:
                raise ValueError("expecting a tensor or a tuple of tensors")
            assert torch.is_tensor(first_input)
            part = PartitionedTensor(tensor=first_input.grad,
                                     group=self.grid.get_slice_parallel_group())

            inputs = (part.to_meta(), part.data(), *inputs_grad_tail)

        # XXX Terrible hack
        # Drop the attention mask from the input buffer here. It does not have
        # a grad that needs to be communicated. We free the buffer immediately
        # after, so no need to restore it. The receiver also has a hack that skips
        # the recv. This is because NCCL does not let us send torch.BoolTensor :-(.
        if self.has_attention_mask or self.has_bool_tensors:
            inputs = list(inputs)
            inputs.pop()
            inputs = tuple(inputs)

        def send_handler(stage):
            if isinstance(inputs, torch.Tensor):
                assert inputs.grad is not None
                p2p.send(inputs.grad, stage)
            else:
                # XXX terrible hacky branch
                if self.is_grad_partitioned:
                    # First two sends are partitioned gradient
                    p2p.send(inputs[0], stage)
                    p2p.send(inputs[1], stage)
                else:
                    for idx, buffer in enumerate(inputs):
                        # Skip tensors that will not produce a grad
                        if not buffer.is_floating_point():
                            assert buffer.grad is None
                            continue
                        assert buffer.grad is not None
                        p2p.send(buffer.grad, stage)

        try:
            send_handler(self.prev_stage)
        except Exception as e:
            raise
            dp_id = self.grid.get_data_parallel_id()
            failed_rank = self.grid._topo.get_rank(data=dp_id, pipe=self.prev_stage)
            self.global_store.set(f'exception-{failed_rank}', '1')
            self.log(f'---- STEP {self.global_steps}, SEND GRADS TO {failed_rank} '
                     f'(pipe={self.prev_stage}, data={dp_id}) FAILED!!!! RESORTING TO FALLBACK {e}', color='r')
            raise PrevStageException(e, dp_id=dp_id, stage_id=self.prev_stage)

        # We can free up the input buffer now
        self.pipe_buffers['inputs'][buffer_id] = None

        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').stop()

    def _exec_recv_activations(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').start()

        def recv_handler(stage):
            recvd = None

            # Allocate the buffer if necessary
            if self.pipe_recv_buf is None:
                self.pipe_recv_buf = self._recv_tensor_meta(stage)

            if isinstance(self.pipe_recv_buf, torch.Tensor):
                p2p.recv(self.pipe_recv_buf, stage)
                recvd = self.pipe_recv_buf.clone().detach()
                recvd.requires_grad = recvd.is_floating_point()
            else:
                assert isinstance(self.pipe_recv_buf, tuple)
                recvd = [None] * len(self.pipe_recv_buf)
                for idx, buffer in enumerate(self.pipe_recv_buf):
                    assert torch.is_tensor(buffer)
                    # XXX hardcode meta type
                    if self.is_pipe_partitioned and idx == 0 and buffer.dtype != torch.long:
                        if self.meta_buffer is None:
                            self.meta_buffer = torch.zeros(buffer.size(),
                                                           dtype=torch.long,
                                                           device=self.device)
                        buffer = self.meta_buffer

                    p2p.recv(buffer, stage)
                    recvd[idx] = buffer.clone().detach()

                # NCCL does not like to send torch.BoolTensor types, so un-cast the
                # attention mask
                if self.has_attention_mask or self.has_bool_tensors:
                    recvd[-1] = recvd[-1].bool()

                recvd = tuple(recvd)

                for buffer in recvd:
                    buffer.requires_grad = buffer.is_floating_point()

            self.pipe_buffers['inputs'][buffer_id] = recvd

        try:
            recv_handler(self.prev_stage)
        except Exception as e:
            raise
            dp_id = self.grid.get_data_parallel_id()
            failed_rank = self.grid._topo.get_rank(data=dp_id, pipe=self.prev_stage)
            self.global_store.set(f'exception-{failed_rank}', '1')
            self.log(f'---- STEP {self.global_steps}, RECV ACTS FROM {failed_rank} '
                     f'(pipe={self.prev_stage}, data={dp_id}) FAILED!!!! RESORTING TO FALLBACK {e}', color='r')
            raise PrevStageException(e, dp_id=dp_id, stage_id=self.prev_stage)

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').stop()

    def _exec_recv_grads(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]
        # XXX these shapes are hardcoded for Megatron
        # Restore partitioned output if it was partitioned and we are sending full gradients
        if self.is_pipe_partitioned and not self.is_grad_partitioned:
            part_output = PartitionedTensor.from_meta(
                meta=outputs[0],
                local_part=outputs[1],
                group=self.grid.get_slice_parallel_group())
            outputs[0].data = part_output.full()
            outputs = (outputs[0], *outputs[2:])
            # save for backward
            self.pipe_buffers['outputs'][buffer_id] = outputs

        # Allocate gradient if necessary
        if self.grad_layer is None:
            if isinstance(outputs, torch.Tensor):
                s = list(outputs.size())
                self.grad_layer = self._allocate_buffer(s,
                                                        dtype=outputs.dtype,
                                                        num_buffers=1)[0]
            else:
                # XXX This is a HACK
                # When we exchange activations/gradients, the two pipe stages
                # need to issue the send/recv with the same buffer sizes or
                # else there is a deadlock. The is_floating_point() filter is
                # used to avoid sending gradients for tensors that do not
                # produce gradients. When TP>1, we partition the first
                # activations/gradients across TP ranks to save communication
                # volume and memory. That partitioned tensor is represented as
                # two tensors: a 1/TPth chunk of the original data and also a
                # small LongTensor storing the metadata used to reconstruct on
                # the other side. When combined, the floating point filter also
                # filtered out the metadata tensor. This quick (hacky) fix just
                # branches on is_grad_partitioned so we don't filter out the
                # metadata tensor.
                if self.is_grad_partitioned:
                    sizes_and_dtypes = [
                        (list(t.size()),
                         t.dtype) for t in outputs[:2]
                    ] + [(list(t.size()),
                          t.dtype) for t in outputs[2:] if t.is_floating_point()]
                else:
                    sizes_and_dtypes = [(list(t.size()),
                                         t.dtype) for t in outputs
                                        if t.is_floating_point()]
                self.grad_layer = self._allocate_buffers(sizes_and_dtypes,
                                                         num_buffers=1)[0]

        def recv_handler(stage):
            if isinstance(self.grad_layer, torch.Tensor):
                p2p.recv(self.grad_layer, stage)
            else:
                assert isinstance(outputs, tuple)
                for idx, buffer in enumerate(self.grad_layer):
                    # XXX GPT-2 hack
                    if self.is_grad_partitioned and idx == 0 and buffer.dtype != torch.long:
                        buffer.data = torch.zeros(buffer.size(),
                                                  dtype=torch.long,
                                                  device=self.device)
                    p2p.recv(buffer, stage)

        try:
            recv_handler(self.next_stage)
        except Exception as e:
            raise
            dp_id = self.grid.get_data_parallel_id()
            failed_rank = self.grid._topo.get_rank(data=dp_id, pipe=self.next_stage)
            self.global_store.set(f'exception-{failed_rank}', '1')
            self.log(f'---- STEP {self.global_steps}, RECV GRADS FROM {failed_rank} '
                     f'(pipe={self.next_stage}, data={dp_id}) FAILED!!!! RESORTING TO FALLBACK {e}', color='r')
            raise NextStageException(e, dp_id=dp_id, stage_id=self.next_stage)

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').stop()

    def _exec_optimizer_step(self, lr_kwargs=None):
        if self.wall_clock_breakdown():
            self.timers('step_microstep').start()
            self.timers('step').start()
        self.mem_status('BEFORE STEP', reset_max=True)

        # sync gradient before clipping since clip_grad needs to communicate with model_parallel_group
        # begin to sync redundant gradients for non-redundant nodes
        # if self.enable_redundant_model_states and not self.is_redundant_node and (self.global_steps + 1) % self.redundant_states_sync_interval == 0:
        if self.enable_redundant_model_states and self.is_redundant_sender:
            self.sync_redundant_gradients_wait()

        self._force_grad_boundary = True
        self.last_total_norm = self._take_clip_gradients()

        # prepare redundant gradients for next synchronization
        if self.enable_redundant_model_states and self.is_redundant_sender and (self.global_steps + 1) % self.redundant_states_sync_interval == 0:
            self.sync_redundant_gradients()

        self._take_model_step(lr_kwargs, clip_gradients=False)
        self._force_grad_boundary = False

        self.mem_status('AFTER STEP')

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/lr',
                                    self.get_lr()[0],
                                    self.global_samples)]
            if self.fp16_enabled() and hasattr(self.optimizer, 'cur_scale'):
                self.summary_events.append((f'Train/Samples/loss_scale',
                                            self.optimizer.cur_scale,
                                            self.global_samples))
            self.monitor.write_events(self.summary_events)

        if self.wall_clock_breakdown():
            self.timers('step_microstep').stop()
            self.timers('step').stop()
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    'batch_input',
                    'forward_microstep',
                    'backward_microstep',
                    'backward_inner_microstep',
                    'backward_allreduce_microstep',
                    'backward_tied_allreduce_microstep',
                    'step_microstep'
                ])
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    'forward',
                    'backward',
                    'backward_inner',
                    'backward_allreduce',
                    'step'
                ])

    def _zero_grads(self, inputs):
        if isinstance(inputs, torch.Tensor):
            if inputs.grad is not None:
                inputs.grad.data.zero_()
        else:
            for t in inputs:
                if t.grad is not None:
                    t.grad.data.zero_()

    def _allocate_zeros(self, shape, **kwargs):
        """ Allocate a tensor of zeros on the engine's device.

        Arguments:
            shape: the shape of the tensor to allocate
            kwargs: passed to torch.zeros()

        Returns:
            A tensor from torch.zeros() allocated on self.device.
        """
        if "dtype" not in kwargs:
            if self.fp16_enabled():
                kwargs["dtype"] = torch.half
            if self.bfloat16_enabled():
                kwargs["dtype"] = torch.bfloat16

        return torch.zeros(shape, device=self.device, **kwargs)

    def _allocate_buffer(self, shape, num_buffers=-1, **kwargs):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffers.append(self._allocate_zeros(shape, **kwargs))
        return buffers

    def _allocate_buffers(self, shapes_and_dtypes, requires_grad=False, num_buffers=-1):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffer = []
            for shape, dtype in shapes_and_dtypes:
                buffer.append(
                    self._allocate_zeros(shape,
                                         dtype=dtype,
                                         requires_grad=requires_grad))
            buffers.append(buffer)
        return buffers

    def forward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def backward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def step(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def report_memory(self, force=False):
        """Simple GPU memory report."""
        if self.report_memory_flag or force:
            mega_bytes = 1024.0 * 1024.0
            string = f'rank {self.global_rank} stage {self.stage_id} memory (MB)'
            string += ' | allocated: {:.3f}'.format(
                torch.cuda.memory_allocated() / mega_bytes)
            string += ' | max allocated: {:.3f}'.format(
                torch.cuda.max_memory_allocated() / mega_bytes)
            string += ' | reserved: {:.3f}'.format(torch.cuda.memory_reserved() / mega_bytes)
            string += ' | max reserved: {:.3f}'.format(
                torch.cuda.max_memory_reserved() / mega_bytes)
            if self.grid.data_parallel_id == 0 or force:
                self.log(string, color='g')
            if not force:
                self.report_memory_flag = False

    def mem_status(self, msg, print_rank=-1, reset_max=False):
        return
        global mem_alloced, mem_cached
        if not self.global_steps == 0 or not self.global_steps == 9:
            #return
            pass
        if self.mpu.get_data_parallel_rank() != 0:
            return

        if self.global_rank != 0:
            return

        rank = self.global_rank
        if print_rank != -1 and rank != print_rank:
            return

        torch.cuda.synchronize()

        if reset_max:
            torch.cuda.reset_max_memory_cached()
            torch.cuda.reset_max_memory_allocated()

        new_alloced = torch.cuda.memory_allocated()
        new_cached = torch.cuda.memory_cached()

        delta_alloced = new_alloced - mem_alloced
        delta_cached = new_cached - mem_cached

        mem_cached = new_cached
        mem_alloced = new_alloced

        max_alloced = torch.cuda.max_memory_allocated()
        max_cached = torch.cuda.max_memory_cached()

        # convert to GB for printing
        new_alloced /= 1024**3
        new_cached /= 1024**3
        delta_alloced /= 1024**3
        delta_cached /= 1024**3
        max_alloced /= 1024**3
        max_cached /= 1024**3

        print(
            f'RANK={rank} STAGE={self.stage_id} STEP={self.global_steps} MEMSTATS',
            msg,
            f'current alloc={new_alloced:0.4f}GB (delta={delta_alloced:0.4f}GB max={max_alloced:0.4f}GB) '
            f'current cache={new_cached:0.4f}GB (delta={delta_cached:0.4f}GB max={max_cached:0.4f}GB)'
        )

    def module_state_dict(self):
        """Override hack to save a pipe model and return the directory path of the save.

        This method should only be called by DeepSpeed's ``save_checkpoint()``. The
        recommended way of saving a ``PipelineModule`` outside of ``save_checkpoint()``
        is ``save_state_dict()``.

        Returns:
            None
        """
        assert isinstance(self.module, PipelineModule)
        assert self._curr_ckpt_path is not None, \
            "PipelineEngine expects module_state_dict() to be called from save_checkpoint()"

        self.module.save_state_dict(self._curr_ckpt_path,
                                    checkpoint_engine=self.checkpoint_engine)
        return None

    def load_module_state_dict(self, state_dict, strict=True, custom_load_fn=None):
        """Override hack to instead use a directory path.

        This is important because pipeline models checkpoint by layer instead of rank.

        If ``state_dict`` is not ``None`` or a ``str``, we revert to ``super()`` expecting a ``dict``.

        Args:
            state_dict (str, None): unused
            strict (bool, optional): Strict state loading. Defaults to True.
        """
        assert custom_load_fn is None, "custom_load_fn not supported w. pipeline parallelism"
        if (state_dict is not None) and (not isinstance(state_dict, str)):
            super().load_module_state_dict(state_dict, strict)
            return

        self.module.load_state_dir(load_dir=self._curr_ckpt_path,
                                   strict=strict,
                                   checkpoint_engine=self.checkpoint_engine)

    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    _INSTRUCTION_MAP = {
        schedule.OptimizerStep: _exec_optimizer_step,
        schedule.ReduceGrads: _exec_reduce_grads,
        schedule.ReduceTiedGrads: _exec_reduce_tied_grads,
        schedule.LoadMicroBatch: _exec_load_micro_batch,
        schedule.ForwardPass: _exec_forward_pass,
        schedule.BackwardPass: _exec_backward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
        schedule.SendGrad: _exec_send_grads,
        schedule.RecvGrad: _exec_recv_grads,
    }

    def _exec_schedule(self, pipe_schedule):
        # Reserve and reset buffers.
        self._reserve_pipe_buffers(pipe_schedule.num_pipe_buffers())
        self.fwd_outputs = []

        # begin to sync redundant gradients for non-redundant nodes
        if self.enable_redundant_model_states:
            if self.is_redundant_sender:
                self.send_redundant_gradients()
            if self.is_redundant_node and self.is_redundant_gradients_ready:
                self.recv_redundant_gradients(self.redundant_gradients_recv_num['warmup_phase'])

        exception_status = None
        # For each step in the schedule
        for i, step_cmds in enumerate(pipe_schedule):
            # For each instruction in the step
            for cmd in step_cmds:
                if type(cmd) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(
                        f'{self.__class__.__name__} does not understand instruction {repr(cmd)}'
                    )

                try:
                    # Equivalent to: self._exec_forward_pass(buffer_id=0)
                    # self.log(f'rank: {self.global_rank} stage {self.stage_id} exec {cmd}, data_iter: {self.data_iterator}', color='lg')
                    self.cur_cmd = cmd
                    self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                    self._exec_instr(**cmd.kwargs)
                except Exception as e:
                    raise
                    msg = f'{type(cmd)}: {e}'
                    print(msg)
                    e = type(e)(msg, dp_id=e.dp_id, stage_id=e.stage_id)
                    exception_status = (i, e)
                    # break
                    raise e

            if exception_status is not None:
                break

        return exception_status

    def log(self, msg, color=None):
        output = f'[{datetime.datetime.now()}] [{self.global_rank:02d}|{self.global_steps:02d}] {msg}'
        if color is not None:
            print_color = Fore.RED
            if color == 'r':
                print_color = Fore.RED
            elif color == 'y':
                print_color = Fore.YELLOW
            elif color == 'b':
                print_color = Fore.BLUE
            elif color == 'lb':
                print_color = Fore.LIGHTCYAN_EX
            elif color == 'g':
                print_color = Fore.GREEN
            elif color == 'lg':
                print_color = Fore.LIGHTGREEN_EX
            print(print_color + output + Fore.RESET, flush=True)
        else:
            print(output, flush=True)

    def debug(self, msg, color=None):
        global debug
        if debug:
            self.log(msg, color=color)

    def send_redundant_gradients_helper(self, stage_id):
        # FIXME(jiangfei): fix to iterate TiedModule parameters
        grad_buckets, grad_buckets_size, flat_grad_tensors = [], [], []

        def append_bucket(bucket, bucket_size):
            grad_buckets.append(bucket)
            grad_buckets_size.append(bucket_size)
            grad_tensor = self.flatten(bucket)
            grad_event = torch.cuda.Event(blocking=True)
            grad_event.record()
            flat_grad_tensors.append((grad_tensor, grad_event))

        def post_handle(i):
            flat_g = flat_grad_tensors.pop(0)
            del flat_g

        bucket, bucket_size = [], 0
        for param in self.module.redundant_parameters(stage_id):
            if param.grad is not None:
                if self.fp16_enabled():
                    bucket.append(param.grad.half())
                else:
                    bucket.append(param.grad)
                bucket_size += param.numel()

            if bucket_size >= self.sync_redundant_model_states_bucket_size:
                append_bucket(bucket, bucket_size)
                bucket, bucket_size = [], 0
        if bucket_size > 0:
            append_bucket(bucket, bucket_size)
        # make sure all gradients are ready
        torch.cuda.synchronize()
        del grad_buckets

        with torch.cuda.stream(self.sync_redundant_grad_stream):
            for (grad_tensor, grad_event) in flat_grad_tensors:
                # self.sync_redundant_grad_stream.wait_event(grad_event)
                # grad_event.synchronize()

                grad_tensor = grad_tensor.to('cpu', non_blocking=True)
                send_info = (grad_tensor, post_handle)
                self.redundant_gradients_senders.append(send_info)

    def send_redundant_gradients(self):
        if len(self.redundant_gradients_senders) > 0:
            send_num = self.redundant_gradients_send_num.pop(0)
            for _ in range(send_num):
                grad_tensor, post_handle = self.redundant_gradients_senders.pop(0)
                if self.is_redundant_sender:
                    handle = p2p.isend(grad_tensor, self.num_stages - 1, group=self.grid.gloo_pg)
                    self.sync_redundant_gradients_handlers.append((handle, post_handle))

                post_handle(0)

        # while len(self.redundant_gradients_senders) > 0:
        #     grad_tensor, post_handle = self.redundant_gradients_senders.pop(0)
        #     if self.is_redundant_sender:
        #         handle = p2p.isend(grad_tensor, self.num_stages - 1, group=self.grid.gloo_pg)
        #         self.sync_redundant_gradients_handlers.append((handle, post_handle))

        #     post_handle(0)

    def finalize_send_redundant_gradients(self):
        while len(self.redundant_gradients_senders) > 0:
            grad_tensor, post_handle = self.redundant_gradients_senders.pop(0)
            if self.is_redundant_sender:
                handle = p2p.isend(grad_tensor, self.num_stages - 1, group=self.grid.gloo_pg)
                self.sync_redundant_gradients_handlers.append((handle, post_handle))

            post_handle(0)

    def recv_redundant_gradients_helper(self, stage_id):
        param_buckets, grad_buckets, grad_buckets_size, flat_grad_tensors = [], [], [], []

        def append_bucket(param_bucket, bucket, bucket_size):
            param_buckets.append(param_bucket)
            grad_buckets.append(bucket)
            grad_buckets_size.append(bucket_size)
            grad_tensor = self.flatten(bucket)
            flat_grad_tensors.append(grad_tensor)

        def post_handle(i):
            recv_grads = self.unflatten(flat_grad_tensors[i], grad_buckets[i])
            for param in param_buckets[i]:
                param.grad = recv_grads.pop(0).float()

        with torch.cuda.stream(self.sync_redundant_grad_stream):
            param_bucket, bucket, bucket_size = [], [], 0
            for param in self.module.redundant_parameters(stage_id):
                if param.requires_grad:
                    param_bucket.append(param)
                    if self.fp16_enabled():
                        bucket.append(torch.zeros_like(param, dtype=torch.float16))
                    else:
                        bucket.append(torch.zeros_like(param))
                    bucket_size += param.numel()

                if bucket_size >= self.sync_redundant_model_states_bucket_size:
                    append_bucket(param_bucket, bucket, bucket_size)
                    param_bucket, bucket, bucket_size = [], [], 0
            if bucket_size > 0:
                append_bucket(param_bucket, bucket, bucket_size)

            for i in range(len(flat_grad_tensors)):
                recv_info = (flat_grad_tensors[i], stage_id, post_handle, i)
                self.redundant_gradients_receivers.append(recv_info)

    def recv_redundant_gradients(self, num_receivers):
        for _ in range(num_receivers):
            flat_g, stage_id, post_handle, idx = self.redundant_gradients_receivers.pop(0)
            handle = p2p.irecv(flat_g, stage_id, group=self.grid.gloo_pg)
            self.sync_redundant_gradients_handlers[stage_id].append((handle, post_handle, idx))

    def finalize_recv_redundant_gradients(self):
        while len(self.redundant_gradients_receivers) > 0:
            flat_g, stage_id, post_handle, idx = self.redundant_gradients_receivers.pop(0)
            handle = p2p.irecv(flat_g, stage_id, group=self.grid.gloo_pg)
            self.sync_redundant_gradients_handlers[stage_id].append((handle, post_handle, idx))

    def sync_redundant_gradients(self, stages=None):
        if self.is_redundant_node:
            if stages is None:
                stages = range(self.num_stages - 1)
            for stage_id in stages:
                self.sync_redundant_gradients_handlers[stage_id] = []
                self.recv_redundant_gradients_helper(stage_id)

            # arrange redundant gradients synchronization order
            # n_warmup = self.redundant_warmup_phase_num
            # n_ending = self.redundant_ending_phase_num
            # remains, final_stage, first_stage = [], [], []
            # while len(self.redundant_gradients_receivers) > 0:
            #     info = self.redundant_gradients_receivers.pop(0)
            #     if info[1] == self.num_stages - 2 and len(final_stage) < n_warmup:
            #         final_stage.append(info)
            #     elif info[1] == 0:
            #         first_stage.append(info)
            #     else:
            #         remains.append(info)
            # self.redundant_gradients_receivers = final_stage + first_stage[:-n_ending] + remains + first_stage[-n_ending:]

            stage_to_receivers = {}
            while len(self.redundant_gradients_receivers) > 0:
                info = self.redundant_gradients_receivers.pop(0)
                if info[1] not in stage_to_receivers:
                    stage_to_receivers[info[1]] = []
                stage_to_receivers[info[1]].append(info)

            for i in range(self.redundant_states_sync_interval):
                for stage_id in stages:
                    recv_num = self.stage_recv_nums[stage_id][i]
                    for _ in range(recv_num):
                        info = stage_to_receivers[stage_id].pop(0)
                        self.redundant_gradients_receivers.append(info)

            self.is_redundant_gradients_ready = True

            self.prepare_redundant_recv_nums(len(self.redundant_gradients_receivers))
        else:
            self.sync_redundant_gradients_handlers = []
            self.redundant_gradients_senders = []
            self.send_redundant_gradients_helper(self.stage_id)

            self.prepare_redundant_send_nums(len(self.redundant_gradients_senders))

    def sync_redundant_gradients_wait(self):
        if len(self.sync_redundant_gradients_handlers) <= 0:
            return

        # wait for sync to finish
        with torch.cuda.stream(self.sync_redundant_grad_stream):
            if self.is_redundant_node:
                for _, handlers in self.sync_redundant_gradients_handlers.items():
                    for (handle, post_handle, i) in handlers:
                        handle.wait()
                        post_handle(i)
                self.sync_redundant_gradients_handlers = {}
                for stage_id in range(self.num_stages - 1):
                    self.sync_redundant_gradients_handlers[stage_id] = []
                self._redundant_step_boundary = ((self.global_steps + 1) % self.redundant_states_sync_interval == 0)
            else:
                for i, (handle, post_handle) in enumerate(self.sync_redundant_gradients_handlers):
                    handle.wait()
                    # post_handle(i)
                self.sync_redundant_gradients_handlers = []

    def finalize_sync(self):
        if self.is_redundant_node:
            self.finalize_recv_redundant_gradients()
            self.sync_redundant_gradients_wait()
            self._redundant_step_boundary = True
            self.redundant_step()
        else:
            self.finalize_send_redundant_gradients()
            self.sync_redundant_gradients_wait()

    def build_redundant_optimizer(self):
        params = []
        for stage_id in range(self.num_stages - 1):
            for param in self.module.redundant_parameters(stage_id):
                params.append(param)

        self._redundant_step_boundary = False

        if self.optimizer_name() in ['FusedAdam', 'adam']:
            if self.fp16_enabled():
                optimizer_cfg = self.optimizer.optimizer.defaults.copy()
            else:
                optimizer_cfg = self.optimizer.defaults.copy()
            self.redundant_optimizer = DeepSpeedCPUAdam(params, **optimizer_cfg)
        else:
            raise ValueError('Unsupported redundant optimizer: %s' % self.optimizer_name())

        if self.redundant_model_only:
            # init optimizer state
            for param in params:
                param.grad = torch.zeros_like(param)
            self.redundant_optimizer.step()
            for param in params:
                param.grad = None

        if self.lr_scheduler.__class__.__name__ == 'AnnealingLR':
            self.redundant_lr_scheduler = self.lr_scheduler.__class__(
                self.redundant_optimizer,
                start_lr=self.lr_scheduler.start_lr,
                warmup_iter=self.lr_scheduler.warmup_iter,
                total_iters=self.lr_scheduler.end_iter,
                decay_style=self.lr_scheduler.decay_style,
                last_iter=self.lr_scheduler.num_iters,
                min_lr=self.lr_scheduler.min_lr,
                use_checkpoint_lr_scheduler=self.lr_scheduler.use_checkpoint_lr_scheduler,
                override_lr_scheduler=self.lr_scheduler.override_lr_scheduler
            )
        elif self.lr_scheduler.__class__.__name__ == 'WarmupDecayLR':
            self.redundant_lr_scheduler = self.lr_scheduler.__class__(
                self.redundant_optimizer,
                self.lr_scheduler.total_num_steps,
                warmup_min_lr=0.0,
                warmup_max_lr=0.001,
                warmup_num_steps=10,
                warmup_type='log',
                last_batch_iteration=self.lr_scheduler.last_batch_iteration,
            )
        else:
            raise ValueError('Unsupported redundant lr scheduler: %s' % self.lr_scheduler.__class__.__name__)

        del params

    def redundant_step(self):
        if not self._redundant_step_boundary:
            return
        # clip gradient
        # if self.gradient_clipping() > 0.0:
        #     clip_coef = self.gradient_clipping() / (self.last_total_norm + 1e-6)
        #     if clip_coef < 1:
        #         for stage_id in range(self.num_stages - 1):
        #             for param in self.module.redundant_parameters(stage_id):
        #                 if param.grad is not None:
        #                     param.grad.data.mul_(clip_coef)

        self.redundant_optimizer.step()

        # zero grad
        for stage_id in range(self.num_stages - 1):
            for param in self.module.redundant_parameters(stage_id):
                param.grad = None

        overflow = False
        if hasattr(self.optimizer, "overflow"):
            overflow = self.optimizer.overflow
        if not overflow:
            self.redundant_lr_scheduler.step()

        self._redundant_step_boundary = False
