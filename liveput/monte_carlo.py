import os
import json
import numpy as np

from cost import INF, get_costmodel


STORE = None
GPUS_PER_NODE = int(os.environ.get('GPUS_PER_NODE', 1))


def setup_store(store):
    global STORE
    STORE = store


def get_store():
    global STORE
    return STORE


class MonteCarloLiveput:
    def __init__(self, model, N, D, P, n_in, n_out, store=None, n_sim=1000):
        self.model = model
        self.N = N
        self.D = D
        self.P = P
        self.n_in = n_in
        self.n_out = n_out
        self.store = store
        self.n_sim = n_sim
        self.apply_optimization = True #False

        # partition model
        self.model.partition(P, update=True)

        # target strategies in next interval
        self.target_strategies = {}

        min_pp = 1
        if self.model.name == 'gpt-1.5b':
            min_pp = 6
            max_pp = 32
            ignore_pp = []
            ignore_dp = []
        elif self.model.name == 'gpt-6.7b':
            min_pp = 12
            max_pp = 32
            ignore_pp = [17, 19, 21, 23, 25, 27, 28, 29, 30, 31, 32]
            ignore_dp = []
        elif self.model.name == 'bert':
            min_pp = 2
            max_pp = 16
            ignore_pp = [9, 11, 13, 15]
            ignore_dp = []
        elif self.model.name == 'vgg19':
            min_pp = 2
            max_pp = 9
            ignore_pp = []
            ignore_dp = [1]
        elif self.model.name == 'resnet152':
            min_pp = 2
            max_pp = 16
            ignore_pp = [3, 5, 7, 9, 11, 13, 15]
            ignore_dp = []

        nnodes = N + n_in - n_out
        for dp in range(1, nnodes + 1):
            for pp in range(1, nnodes + 1):
                if dp * pp <= nnodes and pp >= min_pp and pp <= max_pp and pp not in ignore_pp and dp not in ignore_dp:
                    self.target_strategies[(dp, pp)] = []

    def simulate(self):
        for _ in range(self.n_sim):
            self.draw()

        for key in self.target_strategies:
            probability = len(self.target_strategies[key]) / self.n_sim
            if len(self.target_strategies[key]) > 0:
                migration_cost = np.mean(self.target_strategies[key])
            else:
                migration_cost = 0

            if self.store is not None:
                store_key = (self.N, self.D, self.P, self.n_in, self.n_out)
                self.store.update(store_key, {key: (probability, migration_cost)})

    def draw(self):
        # N nodes is encoded as
        # DxP grid: [0,             1, ...,   P-1]
        #           [P,           P+1, ...,  2P-1]
        #           ...
        #           [(D-1)P, (D-1)P+1, ..., D*P-1]
        # candidate pool: D*P, ..., N-1
        preempted_nodes = np.random.choice(range(self.N), self.n_out, replace=False)
        for (dp, pp) in self.target_strategies:
            cost = self.try_migration(preempted_nodes, dp, pp)
            # if cost is INF, it means that the migration is not feasible
            if cost < INF:
                self.target_strategies[(dp, pp)].append(cost)

    def try_migration(self, preempted_nodes, dp, pp):
        # FIXME: we need to discuss when to add new nodes into current training job
        migration_cost, repartition_cost = INF, INF
        if pp == self.P:
            migration_cost = self.migration(preempted_nodes, dp, pp)
        else:
            repartition_cost = self.repartition(preempted_nodes, dp, pp)
        cost = min(migration_cost, repartition_cost)
        return cost

    def estimate(self, preempted_nodes, dp, pp, nsim):
        costs = []
        for _ in range(nsim):
            cost = self.try_migration(preempted_nodes, dp, pp)
            if cost < INF:
                costs.append(cost)
        return np.mean(costs)

    def migration(self, preempted_nodes, dp, pp):
        if len(preempted_nodes) == 0 and dp == self.D and pp == self.P:
            return 0

        remain_stage_replicas = {stage_id: self.D for stage_id in range(self.P)}
        for node_id in preempted_nodes:
            if node_id >= self.D * self.P:
                continue
            stage_id = node_id % self.P
            remain_stage_replicas[stage_id] -= 1

        cost = INF
        if min(remain_stage_replicas.values()) >= dp:
            # first seek to do intra-stage migration
            cost = self.estimate_intra_stage_migration(dp, pp)
        else:
            # then seek to do inter-stage migration
            required_stage_replicas = []
            for stage_id in range(pp):
                required_stage_replicas.append(max(dp - remain_stage_replicas[stage_id], 0))
            cost = self.estimate_inter_stage_migration(required_stage_replicas, dp, pp)
        return cost

    def estimate_intra_stage_migration(self, dp, pp):
        cost = get_costmodel().estimate_intra_stage_migration(dp, pp)
        return cost

    def estimate_inter_stage_migration(self, required_stage_replicas, dp, pp):
        stage_cost, redundant_cost = 0, 0
        for stage_id, nnodes in enumerate(required_stage_replicas):
            if nnodes == 0:
                continue
            volume = self.model.stage_model_states(stage_id) * 4 # in bytes
            cur_stage_cost = get_costmodel().p2p_cost(volume)
            if nnodes == dp or (stage_id == pp - 1 and nnodes == dp - 1):
                redundant_cost += cur_stage_cost
            stage_cost = max(stage_cost, cur_stage_cost)

        cost = max(stage_cost, redundant_cost)
        if self.apply_optimization:
            if dp * pp > self.D * self.P - self.n_out:
                cost += get_costmodel().estimate_restart_cost(dp, pp, type='add')
            else:
                cost += get_costmodel().estimate_restart_cost(dp, pp, type='remove')
        else:
            cost += get_costmodel().estimate_restart_cost(dp, pp, type='add')
        return cost

    def repartition(self, preempted_nodes, dp, pp):
        # FIXME: currently, we adopt a simple way to do repartition.
        # This is not the same with the implementation.
        old_parts = self.model.parts
        new_parts, new_part_params = self.model.partition(pp)

        cost = 0
        if self.apply_optimization:
            node_to_layers = {}
            for node_id in range(self.D * self.P):
                stage_id = node_id % self.P
                node_to_layers[node_id] = set(range(old_parts[stage_id], old_parts[stage_id + 1]))
            for node_id in preempted_nodes:
                if node_id in node_to_layers:
                    del node_to_layers[node_id]

            total_params_need = 0
            for stage_id in range(pp):
                need_layers = set(range(new_parts[stage_id], new_parts[stage_id + 1]))
                best_node_id, num_params = None, INF
                for node_id in node_to_layers:
                    layers = list(need_layers.difference(node_to_layers[node_id]))
                    extra_params = self.model.get_layers_params(layers)
                    if extra_params < num_params:
                        best_node_id, num_params = node_id, extra_params
                if best_node_id is not None:
                    del node_to_layers[best_node_id]
                    total_params_need = num_params

            volume = total_params_need * (1 + self.model.optim_state_cnt) * 4 # in bytes
            cost = get_costmodel().p2p_cost(volume)
            # broadcast cost
            if dp > 1:
                bcost = 0
                for params in new_part_params:
                    volume = params * (1 + self.model.optim_state_cnt) * 4 # in bytes
                    part_cost = get_costmodel().p2p_cost(volume)
                    if part_cost > bcost:
                        bcost = part_cost
                cost += bcost
        else:
            for params in new_part_params:
                volume = params * (1 + self.model.optim_state_cnt) * 4 # in bytes
                cost += get_costmodel().p2p_cost(volume)

        if self.apply_optimization:
            if dp * pp > self.D * self.P - self.n_out:
                cost += get_costmodel().estimate_restart_cost(dp, pp, type='add')
            else:
                cost += get_costmodel().estimate_restart_cost(dp, pp, type='remove')
        else:
            cost += get_costmodel().estimate_restart_cost(dp, pp, type='add')
        return cost


class LiveStore:
    """Liveput monte carlo simulation result store.

    Store format:
        # previous state: (N_0, D_0, P_0), event: (n_in, n_out)
        key: '(N_0, D_0, P_0, n_in, n_out)'
        # value is a dict of simulation result
        value: {
            '(D, P)': (probability, migration_cost),
            ...
        }
    """
    def __init__(self, store_dir, model, restart_cost=None, n_sim=1000, update_cache=False):
        self.store_dir = store_dir
        self.model = model
        self.n_sim = n_sim
        self.update_cache = update_cache
        self.updated_keys = set()
        self.has_new_keys = False

        if restart_cost is None:
            if GPUS_PER_NODE > 1:
                self.filename = f'{self.store_dir}/{self.model.name}-bsz{self.model.train_batch_size}_gpu{GPUS_PER_NODE}.json'
            else:
                self.filename = f'{self.store_dir}/{self.model.name}-bsz{self.model.train_batch_size}.json'
        else:
            self.filename = f'{self.store_dir}/{self.model.name}-rc{restart_cost}-bsz{self.model.train_batch_size}.json'
        self.cache = {}

        os.makedirs(self.store_dir, exist_ok=True)

        self.load()

        min_pp = 1
        if self.model.name == 'gpt-1.5b':
            min_pp = 6
            max_pp = 32
            ignore_pp = []
            ignore_dp = []
        elif self.model.name == 'gpt-6.7b':
            min_pp = 12
            max_pp = 32
            ignore_pp = [17, 19, 21, 23, 25, 27, 28, 29, 30, 31, 32]
            ignore_dp = []
        elif self.model.name == 'bert':
            min_pp = 2
            max_pp = 16
            ignore_pp = [9, 11, 13, 15]
            ignore_dp = []
        elif self.model.name == 'vgg19':
            min_pp = 2
            max_pp = 9
            ignore_pp = []
            ignore_dp = [1]
        elif self.model.name == 'resnet152':
            min_pp = 2
            max_pp = 16
            ignore_pp = [3, 5, 7, 9, 11, 13, 15]
            ignore_dp = []

        self.min_pp = min_pp
        self.max_pp = max_pp
        self.ignore_pp = ignore_pp
        self.ignore_dp = ignore_dp

    def get(self, key):
        # check update
        if self.update_cache and str(key) not in self.updated_keys:
            return None
        if str(key) in self.cache:
            return self.cache[str(key)]
        return None

    def update(self, store_key, item):
        if str(store_key) not in self.cache:
            self.cache[str(store_key)] = {}

        for key, value in item.items():
            self.cache[str(store_key)][str(key)] = value
        self.updated_keys.add(str(store_key))

    def encode_key(self, key):
        return str(key)

    def decode_key(self, key):
        return eval(key)

    def load(self):
        print(f'Load liveput simulation result from {self.filename}')
        try:
            with open(self.filename, 'r') as f:
                cache = json.load(f)
                self.cache.update(cache)
        except:
            print(f'[Warning] No cache file found at {self.filename}')

    def save(self):
        print(f'Save liveput simulation result to {self.filename}')
        with open(self.filename, 'w') as f:
            json.dump(self.cache, f)

    def simulate(self, N, D, P, n_in, n_out):
        mc_sampler = MonteCarloLiveput(self.model, N, D, P, n_in, n_out, store=self, n_sim=self.n_sim)
        mc_sampler.simulate()
        self.has_new_keys = True
