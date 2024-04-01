import random
from cost import INF, get_costmodel


class ThroughputOptimized:
    def __init__(self, model, trace):
        self.model = model
        self.trace = trace
        self.need_checkpoint = True
        self.apply_optimization = False #True

    def optimize(self, ideal=False):
        self.hist = [(0, 0, 0, (0, 0, 0))]
        cur_t = 0
        remain_cost = 0
        N = 0
        prev_strategy = None
        print(f'Throughput optimized approach, ideal={ideal}')
        for interval_id in range(self.trace.num_intervals):
            interval = self.trace.get_next_intervals(interval_id, num_intervals=1)[0]
            nnodes, n_add, n_remove = interval
            if interval_id == 0:
                event = (0, nnodes, 0)
            else:
                event = (N, n_add, n_remove)
            N = nnodes
            strategy = get_costmodel().best_strategy(nnodes)
            speed, throughput = get_costmodel().estimate_strategy_performance(strategy)

            if (cur_t == 0 or n_add + n_remove > 0) and not ideal:
                recfg_type = 'add' if n_add > 0 or cur_t == 0 else 'remove'
                cost = self.reconfiguration(interval_id, nnodes, strategy, prev_strategy, type=recfg_type)
                if remain_cost > 0:
                    # we need to do another reconfiguration at the same time
                    remain_cost = cost + get_costmodel().estimate_concurrent_restart_cost(strategy[0], strategy[1], type=recfg_type)
                else:
                    remain_cost = cost
            else:
                cost = 0

            # throughput: samples/sec,  time: in milli_seconds
            previous_samples = self.hist[-1][1]
            if ideal:
                train_samples = throughput * self.trace.interval_duration
            else:
                train_samples = throughput * max(0, self.trace.interval_duration - remain_cost / 1000)
            total_samples = previous_samples + train_samples

            start_t = min(cur_t + remain_cost, cur_t + self.trace.interval_duration * 1000)
            remain_cost = max(0, remain_cost - self.trace.interval_duration * 1000)
            self.hist.append((start_t, previous_samples, '', event))
            cur_t += self.trace.interval_duration * 1000
            self.hist.append((cur_t, total_samples, strategy, event))
            prev_strategy = strategy
        return self.hist

    def reconfiguration(self, interval_id, nnodes, strategy, prev_strategy, type='no-op'):
        dp, pp = strategy
        new_parts, new_part_params = self.model.partition(pp)

        cost = get_costmodel().estimate_restart_cost(dp, pp, type=type)
        if self.need_checkpoint and interval_id > 0:
            cost += 14_000 # ms for the save and load checkpoint

        comm_cost = 0
        if self.apply_optimization:
            if prev_strategy is not None:
                prev_dp, prev_pp = prev_strategy
                old_parts, _ = self.model.partition(prev_pp)

                if prev_dp * prev_pp > dp * pp:
                    preempted_nodes = random.sample(range(prev_dp * prev_pp), prev_dp * prev_pp - dp * pp)
                else:
                    preempted_nodes = []

                node_to_layers = {}
                for node_id in range(prev_dp * prev_pp):
                    stage_id = node_id % prev_pp
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
                        total_params_need += num_params

                volume = total_params_need * (1 + self.model.optim_state_cnt) * 4 # in bytes
                comm_cost = get_costmodel().p2p_cost(volume)

            # broadcast cost
            if dp > 1:
                bcost = 0
                for params in new_part_params:
                    volume = params * (1 + self.model.optim_state_cnt) * 4 # in bytes
                    part_cost = get_costmodel().p2p_cost(volume)
                    if part_cost > bcost:
                        bcost = part_cost
                comm_cost += bcost
        else:
            for params in new_part_params:
                volume = params * (1 + self.model.optim_state_cnt) * 4 # in bytes
                comm_cost += get_costmodel().p2p_cost(volume)
        cost += comm_cost
        print(f'nnodes: {nnodes}, operation: {type}, strategy: {strategy}, comm_cost: {comm_cost:.3f}, total_cost: {cost:.3f}')
        return cost
