import argparse
import os
import enum
import heapq
import math
import copy
import json
import time

from cost import CostModel, setup_costmodel, get_costmodel, DEFAULT_STRATEGY_CACHE_PATH
from model import ModelSpec, gen_model_signature


parser = argparse.ArgumentParser()
parser.add_argument('--nnodes', type=int, default=32)
parser.add_argument('--ngpu-per-node', type=int, default=1)
# model arch main arguments
parser.add_argument('--model', type=str)
parser.add_argument('--train-batch-size', type=int, default=256)
# model arch optional arguments
parser.add_argument('--nlayers', type=int, default=None)
parser.add_argument('--micro-batch-size', type=int, default=None)
parser.add_argument('--seq-length', type=int, default=None)
# simulator cache
parser.add_argument('--simulation', action='store_true')
parser.add_argument('--update-cache', action='store_true')
args = parser.parse_args()


INF = 1e9
MB = 1024 * 1024
ORACLE = None


class CMD:
    def __init__(self, id):
        self.id = id
        self.finish = False
        self.ready = INF
        self.start = INF

        self.is_steal = False
        self.drop = False

        self.prev = []
        self.next = []

    def add_next(self, cmd):
        self.next.append(cmd)
        cmd.prev.append(self)

    def set_ready(self, tick):
        self.ready = tick

    def set_start(self, tick):
        self.start = tick
        self.end = self.start + self.cost
        self.finish = True

    def free(self):
        for pv in self.prev:
            if not pv.finish:
                return False
        return True

    def __lt__(self, other):
        # return (self.priority, self.ready) < (other.priority, other.ready)
        return (0, self.ready) < (0, other.ready)

    def __repr__(self):
        name = 'F' if self.priority == 1 else 'B'
        return f'{name}{self.id}-{self.ready} ({self.is_steal})'


class ForwardMicroBatchCMD(CMD):
    def __init__(self, id, dp_rank, pp_rank):
        super().__init__(id)
        self.dp_rank = dp_rank
        self.pp_rank = pp_rank
        self.cost = 1
        self.priority = 1

class BackwardMicroBatchCMD(CMD):
    def __init__(self, id, dp_rank, pp_rank):
        super().__init__(id)
        self.dp_rank = dp_rank
        self.pp_rank = pp_rank
        self.cost = 2
        self.priority = 0


class Simulator:
    def __init__(self, model, dp_deg, pp_deg, n_micro_batches, recompute=False):
        self.model = model
        self.dp_deg = dp_deg
        self.pp_deg = pp_deg
        self.n_micro_batches = n_micro_batches
        self.recompute = recompute

        self.stage_p2p_volume = self.model['stage_p2p_volume']

    def stage_comm(self):
        return get_costmodel().p2p_cost(self.stage_p2p_volume * MB)

    def all_reduce(self, param_size, nodes, half=True):
        if half:
            param_size = param_size / 2
        return get_costmodel().all_reduce_cost(param_size * MB, nodes)

    def create_dep(self):
        graphs = {}
        for i in range(self.n_micro_batches):
            for dp_rank in range(self.dp_deg):
                for pp_rank in range(self.pp_deg):
                    fmb = ForwardMicroBatchCMD(i, dp_rank, pp_rank)
                    fmb.cost = self.model['forward'][pp_rank]

                    bmb = BackwardMicroBatchCMD(i, dp_rank, pp_rank)
                    bmb.cost = self.model['backward'][pp_rank]
                    if self.recompute:
                        bmb.cost += self.model['forward'][pp_rank]
                    graphs[(i, dp_rank, pp_rank)] = (fmb, bmb)

        for dp_rank in range(self.dp_deg):
            for pp_rank in range(self.pp_deg):
                for i in range(self.n_micro_batches - 1):
                    graphs[(i, dp_rank, pp_rank)][0].add_next(graphs[(i + 1, dp_rank, pp_rank)][0])
                    graphs[(i, dp_rank, pp_rank)][1].add_next(graphs[(i + 1, dp_rank, pp_rank)][1])

        for i in range(self.n_micro_batches):
            for dp_rank in range(self.dp_deg):
                for pp_rank in range(self.pp_deg):
                    if pp_rank != self.pp_deg - 1:
                        graphs[(i, dp_rank, pp_rank)][0].add_next(graphs[(i, dp_rank, pp_rank + 1)][0])
                        graphs[(i, dp_rank, pp_rank + 1)][1].add_next(graphs[(i, dp_rank, pp_rank)][1])
                    else:
                        graphs[(i, dp_rank, pp_rank)][0].add_next(graphs[(i, dp_rank, pp_rank)][1])

        # max ongoing
        for dp_rank in range(self.dp_deg):
            for pp_rank in range(self.pp_deg):
                ongoing = self.pp_deg - pp_rank
                for i in range(self.n_micro_batches):
                    if i + ongoing < self.n_micro_batches:
                        graphs[(i, dp_rank, pp_rank)][1].add_next(graphs[(i + ongoing, dp_rank, pp_rank)][0])

        self.graphs = graphs

    def schedule(self):
        self.create_dep()

        self.queues = []
        for dp_rank in range(self.dp_deg):
            init = self.graphs[(0, dp_rank, 0)][0]
            init.set_ready(0)
            heapq.heappush(self.queues, init)

        self.pp_ticks = {(dp_rank, pp_rank): 0 for pp_rank in range(self.pp_deg) for dp_rank in range(self.dp_deg)}

        while len(self.queues) > 0:
            batch = heapq.heappop(self.queues)

            start = max(self.pp_ticks[(batch.dp_rank, batch.pp_rank)], batch.ready)
            batch.set_start(start)
            self.pp_ticks[(batch.dp_rank, batch.pp_rank)] = batch.end

            for nxt in batch.next:
                if nxt.free():
                    nxt.set_ready(batch.end + self.stage_comm())
                    heapq.heappush(self.queues, nxt)

        for (dp_rank, pp_rank) in self.pp_ticks:
            dp_deg = (self.dp_deg * 2) if pp_rank == 0 else self.dp_deg
            self.pp_ticks[(dp_rank, pp_rank)] += self.all_reduce(self.model['params'][pp_rank], dp_deg)
            self.pp_ticks[(dp_rank, pp_rank)] += self.model['optimizer'][pp_rank]

        return max(list(self.pp_ticks.values()))


candidate_cache = {}
def candidate_strategy_list(model, ngpus):
    if ngpus in candidate_cache:
        return candidate_cache[ngpus]

    ret = []
    for pp in range(1, ngpus + 1):
        for dp in range(1, ngpus + 1):
            if dp * pp > ngpus:
                continue
            if pp > model.max_pipeline_stages():
                continue
            ret.append((dp, pp))

    candidate_cache[ngpus] = ret
    return ret


class TrainOracle:
    cache = {}

    def __init__(self, model, nnodes, ngpu_per_node, signature, cache_file=DEFAULT_STRATEGY_CACHE_PATH):
        self.model = model
        self.nnodes = nnodes
        self.ngpu_per_node = ngpu_per_node
        self.signature = signature
        self.cache_file = cache_file
        self.key = json.dumps(self.signature)

        # p2p communication size
        self.stage_p2p_volume = self.model.stage_p2p_volume()

        self.candidate_partitions_cache = {}
        self.partition_model_cache = {}
        self.optimal_cache = {}

    @property
    def bw(self):
        # convert to GB/s
        return self.signature['bw'] / 8

    @property
    def micro_batch_size(self):
        return self.signature['micro_batch_size']

    @property
    def train_batch_size(self):
        return self.signature['train_batch_size']

    def candidate_partitions(self, pp):
        if pp in self.candidate_partitions_cache:
            return self.candidate_partitions_cache[pp]

        base_parts = self.model.max_pipeline_stages() // pp
        remain_parts = self.model.max_pipeline_stages() % pp

        ret = []

        # manually assign
        partition = [base_parts] * pp
        for i in range(remain_parts):
            partition[i + 1] += 1
        partition = list(reversed(partition))
        if self.model.has_embedding():
            partition[0] += 1 # embedding layer
        ret.append(partition)

        self.candidate_partitions_cache[pp] = ret
        return self.candidate_partitions_cache[pp]

    def partition_model(self, partition):
        key = tuple(partition)
        if key in self.partition_model_cache:
            return self.partition_model_cache[key]

        stage_forward_speed = [0] * len(partition)
        stage_backward_speed = [0] * len(partition)
        stage_optimizer_speed = [0] * len(partition)
        acts, params = [0] * len(partition), [0] * len(partition)
        recompute_mem = [math.inf] * len(partition)

        for i, nlayer in enumerate(partition):
            if self.model.has_embedding():
                for chunk in range(1, nlayer + 1):
                    act_size = chunk * self.model[1]['acts'] + math.ceil(nlayer / chunk) * self.stage_p2p_volume
                    recompute_mem[i] = min(recompute_mem[i], act_size)
            else:
                recompute_mem[i] = min(recompute_mem[i], self.model[i]['acts'])

        partition = [sum(partition[:i+1]) for i in range(len(partition))]
        cur_part = 0
        for i, stats in enumerate(self.model):
            while i >= partition[cur_part]:
                cur_part += 1
            stage_forward_speed[cur_part] += stats['forward']
            stage_backward_speed[cur_part] += stats['backward']
            stage_optimizer_speed[cur_part] += stats['optimizer']
            acts[cur_part] += stats['acts']
            params[cur_part] += stats['params'] * 4 / MB

        if self.model.has_embedding() and len(partition) == 1:
            params[0] -= self.model[-1]['params'] * 4 / MB

        partitioned_model = {
            'forward': stage_forward_speed,
            'backward': stage_backward_speed,
            'optimizer': stage_optimizer_speed,
            'acts': acts,
            'params': params,
            'recompute_size': recompute_mem,
            'stage_p2p_volume': self.stage_p2p_volume,
        }

        self.partition_model_cache[key] = partitioned_model
        return self.partition_model_cache[key]

    def search_for_optimal(self, nnodes):
        if nnodes in self.optimal_cache:
            return self.optimal_cache[nnodes]

        best_cfg, best_speed = None, INF
        for cfg, value in TrainOracle.cache[self.key].items():
            if cfg[0] * cfg[1] <= nnodes:
                if value[0] <= best_speed:
                    best_speed = value[0]
                    best_cfg = cfg

        self.optimal_cache[nnodes] = best_cfg, best_speed
        return self.optimal_cache[nnodes]

    def train_speed(self, dp, pp):
        return TrainOracle.cache[self.key][(dp, pp)][0]

    def best_partition(self, dp, pp):
        return TrainOracle.cache[self.key][(dp, pp)][1]

    def gen_candidates(self, nnodes):
        ret = []
        for cfg, value in TrainOracle.cache[self.key].items():
            if cfg[0] * cfg[1] <= nnodes:
                ret.append(cfg)
        return ret

    def simulate(self, update_cache=True):
        self.load_cache()
        if self.key in TrainOracle.cache and not update_cache:
            return

        print('begin to simulate all training configs...')
        candidate_configs = candidate_strategy_list(self.model, self.nnodes * self.ngpu_per_node)
        for dp, pp in candidate_configs:
            best_partition, best_speed, best_throughput = None, INF, 0
            for partition in self.candidate_partitions(pp):
                partitioned_model = self.partition_model(partition)

                # check oom
                oom, recompute = False, False
                for stage_id in range(pp):
                    ongoing = pp - stage_id
                    model_states_size = partitioned_model['params'][stage_id] * (self.model.optim_state_cnt + 1)
                    if model_states_size + partitioned_model['recompute_size'][stage_id] > 15.5 * 1024:
                        oom = True
                        break
                    if ongoing * partitioned_model['acts'][stage_id] > 15.5 * 1024:
                        recompute = True

                if oom:
                    speed = INF
                    throughput = 0
                else:
                    n_micro_batches = math.ceil(
                        (self.train_batch_size // self.micro_batch_size) / dp
                    )
                    speed = Simulator(partitioned_model, dp, pp, n_micro_batches, recompute=recompute).schedule()
                    throughput = self.micro_batch_size * dp * n_micro_batches / (speed / 1000)

                if throughput > best_throughput:
                    best_throughput = throughput
                    best_speed = speed
                    best_partition = partition

            if best_speed < INF:
                self.save_conifg(dp, pp, best_partition, best_speed, best_throughput)
            print(f'({dp:2d}, {pp:2d}): ({best_speed}, {best_throughput}),')

        if update_cache:
            self.dump_cache()

    def save_conifg(self, dp, pp, partition, speed, throughput):
        if self.key not in TrainOracle.cache:
            TrainOracle.cache[self.key] = {}
        TrainOracle.cache[self.key][(dp, pp)] = (speed, throughput, partition)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as fp:
                tmp_cache = json.load(fp)
                for signature, oracle in tmp_cache.items():
                    TrainOracle.cache[signature] = {}
                    for cfg, value in oracle.items():
                        dp, pp = cfg.split('-')
                        TrainOracle.cache[signature][(int(dp), int(pp))] = value

    def dump_cache(self):
        with open(self.cache_file, 'w') as fp:
            tmp_cache = {}
            for signature, oracle in TrainOracle.cache.items():
                tmp_cache[signature] = {}
                for (dp, pp), value in oracle.items():
                    tmp_cache[signature][f'{dp}-{pp}'] = value
            json.dump(tmp_cache, fp)


def simulate_training_configs(model, nnodes, ngpu_per_node, simulation=False):
    signature = gen_model_signature(model, nnodes, ngpu_per_node)

    # setup cost model
    costmodel = CostModel(model=model.name, model_signature=signature)
    setup_costmodel(costmodel)

    global ORACLE
    ORACLE = TrainOracle(model, nnodes, ngpu_per_node, signature, cache_file=DEFAULT_STRATEGY_CACHE_PATH)
    if simulation:
        ORACLE.simulate(update_cache=args.update_cache)
    if args.update_cache:
        ORACLE.load_cache()


if __name__ == '__main__':
    optional_args = {}
    if args.micro_batch_size is not None:
        optional_args['micro_batch_size'] = args.micro_batch_size
    if args.nlayers is not None:
        optional_args['nlayers'] = args.nlayers
    if args.seq_length is not None:
        optional_args['seq_length'] = args.seq_length

    # TODO: support multi-gpu instances
    assert args.ngpu_per_node == 1

    model = ModelSpec.build(args.model, args.train_batch_size, **optional_args)
    simulate_training_configs(model, args.nnodes, args.ngpu_per_node, simulation=args.simulation)
