import os
import json
import time
from cost import get_costmodel, INF
from monte_carlo import get_store


class LiveputOptimized:
    def __init__(self, model, trace):
        self.model = model
        self.trace = trace
        self.interval_to_overhead = {}

    def optimize(self, look_ahead=1, predict=False):
        print(f'Liveput optimized approach, look ahead={look_ahead}, predict={predict}')
        # hist element: (time, total samples, strategy, interval event)
        hist = [(0, 0, 0, (0, 0, 0))]
        cur_t = 0

        interval_id = -1
        N = self.trace.get_node_number(-1)
        D = P = status = 0
        remain_cost = 0
        while True:
            if interval_id >= self.trace.num_intervals - 1:
                break

            start_time = time.time()
            future_intervals = self.trace.get_next_intervals(interval_id + 1, num_intervals=look_ahead, predict=predict)
            if predict:
                if self.model.name == 'gpt-6.7b':
                    find_low = False
                    new_intervals = []
                    for interval in future_intervals:
                        prev_N, n_in, n_out = interval
                        # hard contraint for min-pp
                        if prev_N + + n_in - n_out < 12:
                            new_intervals.append([12, 0, 0])
                        else:
                            new_intervals.append(interval)
                    future_intervals = new_intervals

            strategy, prob, cost = self.solve_look_ahead(N, D, P, status, remain_cost, future_intervals)
            prev_D, prev_P = D, P

            interval_id += 1
            D, P, _ = strategy
            next_interval = self.trace.get_next_intervals(interval_id, num_intervals=1, predict=False)[0]
            if interval_id == 0:
                event = (0, next_interval[0], 0)
            else:
                event = (N, next_interval[1], next_interval[2])
            if next_interval[0] < D * P:
                D = next_interval[0] // P
                if D == 0:
                    D = 1
                    P = next_interval[0]

            if predict:
                livestore_res = get_store().get((N, prev_D, prev_P, next_interval[1], next_interval[2]))
                try:
                    if livestore_res is None:
                        get_store().simulate(N, prev_D, prev_P, next_interval[1], next_interval[2])
                        livestore_res = get_store().get((N, prev_D, prev_P, next_interval[1], next_interval[2]))
                    prob, cost = livestore_res[get_store().encode_key((D, P))]
                except:
                    print(f'Key: {(N, prev_D, prev_P, next_interval[1], next_interval[2])}, live_res: {livestore_res}')
                    raise
                print(f'interval {interval_id}: {strategy}->{(D, P)}, {prob:.3f}, {cost:.3f}, event: {event}, future: {future_intervals}')
                strategy = (D, P)
            else:
                print(f'interval {interval_id}: {strategy}, {prob:.3f}, {cost:.3f}, event: {event}, future: {future_intervals}')

            N = next_interval[0]

            speed, throughput = get_costmodel().estimate_strategy_performance((D, P))
            if cost > 0:
                if remain_cost > 0:
                    remain_cost = cost + get_costmodel().estimate_concurrent_restart_cost(D, P)
                else:
                    remain_cost = cost
            work_time = self.trace.interval_duration * 1000 - remain_cost

            # throughput: samples/sec,  time: in milli_seconds
            previous_samples = hist[-1][1]
            if work_time >= 0:
                train_samples = throughput * prob * work_time / 1000
                status = 0
            else:
                train_samples = 0
                status = 1
            total_samples = previous_samples + train_samples

            start_t = min(cur_t + remain_cost, cur_t + self.trace.interval_duration * 1000)
            remain_cost = max(0, -work_time)
            hist.append((start_t, previous_samples, '', event))
            cur_t += self.trace.interval_duration * 1000
            hist.append((cur_t, total_samples, strategy, event))

            end_time = time.time()
            self.interval_to_overhead[interval_id] = end_time - start_time

        # save overhead time
        if os.environ.get('OPT_TIME_FILE', None):
            filename = os.environ['OPT_TIME_FILE']
            if predict:
                filename += '_pred.json'
            else:
                filename += '_ideal.json'
            print(f'Saving optimization time stats into {filename}')
            with open(f'{filename}', 'w') as fp:
                json.dump(self.interval_to_overhead, fp)
        return hist

    def look_ahead_one_interval(self, DPF, prev, interval_idx, N, D, P, status, n_in, n_out):
        key = (N, D, P, n_in, n_out)
        if get_store().get(key) is None:
            print(f'>>>> {key} not found, simulating...')
            get_store().simulate(N, D, P, n_in, n_out)

        if interval_idx not in DPF:
            DPF[interval_idx] = {}
            prev[interval_idx] = {}

        for strategy, (p, c) in get_store().get(key).items():
            (dp, pp) = get_store().decode_key(strategy)
            if pp < get_store().min_pp or pp > get_store().max_pp or pp in get_store().ignore_pp:
                continue
            speed, throughput = get_costmodel().estimate_strategy_performance((dp, pp))

            prev_dpf_key = (D, P, status)
            if status:
                # in migration status
                if c > 0:
                    total_cost = c + get_costmodel().estimate_concurrent_restart_cost(dp, pp)
                else:
                    total_cost = DPF[interval_idx - 1][prev_dpf_key][1]
            else:
                total_cost = c
            # remaining time in seconds
            remain_cost = self.trace.interval_duration * p - total_cost / 1000

            if status:
                prev_dpf_samples = DPF[interval_idx - 1][prev_dpf_key][0]
            else:
                prev_dpf_samples = DPF[interval_idx - 1][prev_dpf_key]

            if remain_cost >= 0:
                train_samples = throughput * remain_cost

                dpf_key = (dp, pp, 0)
                if dpf_key not in DPF[interval_idx]:
                    DPF[interval_idx][dpf_key] = 0

                if prev_dpf_samples + train_samples > DPF[interval_idx][dpf_key]:
                    DPF[interval_idx][dpf_key] = prev_dpf_samples + train_samples
                    prev[interval_idx][dpf_key] = prev_dpf_key
                elif prev_dpf_samples + train_samples == DPF[interval_idx][dpf_key]:
                    if dpf_key in prev[interval_idx]:
                        origin_prev_dpf_key = prev[interval_idx][dpf_key]
                        # prefer shallow pipeline depth.
                        # if origin_prev_dpf_key[1] > prev_dpf_key[1] and prev_dpf_key[1] != 1:
                        if origin_prev_dpf_key[1] > prev_dpf_key[1]:
                            prev[interval_idx][dpf_key] = prev_dpf_key

            else:
                dpf_key = (dp, pp, 1)
                if dpf_key not in DPF[interval_idx]:
                    DPF[interval_idx][dpf_key] = (0, INF)

                remain_cost = - remain_cost * 1000 # in milli_seconds
                if prev_dpf_samples > DPF[interval_idx][dpf_key][0] or (
                    prev_dpf_samples == DPF[interval_idx][dpf_key][0] and remain_cost < DPF[interval_idx][dpf_key][1]
                ):
                    DPF[interval_idx][dpf_key] = (prev_dpf_samples, remain_cost)
                    prev[interval_idx][dpf_key] = prev_dpf_key

    def solve_look_ahead(self, N, D, P, cur_status, cur_remain_cost, future_intervals):
        DPF = {}
        prev = {}
        # initialize
        if cur_status:
            DPF[0] = {(D, P, cur_status): (0, cur_remain_cost)}
            prev[0] = {(D, P, cur_status): None}
        else:
            DPF[0] = {(D, P, cur_status): 0}
            prev[0] = {(D, P, cur_status): None}
        nnodes = N
        for interval_idx, interval in enumerate(future_intervals):
            _, n_in, n_out = interval
            for (dp, pp, status), _ in DPF[interval_idx].items():
                self.look_ahead_one_interval(DPF, prev, interval_idx + 1, nnodes, dp, pp, status, n_in, n_out)
            nnodes = nnodes + n_in - n_out

        # reverse
        best_total_samples, best_remain_cost = 0, INF
        for (dp, pp, status), _ in DPF[len(future_intervals)].items():
            if status:
                total_samples, remain_cost = DPF[len(future_intervals)][(dp, pp, status)]
            else:
                total_samples = DPF[len(future_intervals)][(dp, pp, status)]
                remain_cost = 0

            if total_samples > best_total_samples or (
                total_samples == best_total_samples and remain_cost < best_remain_cost
            ):
                best_total_samples = total_samples
                best_remain_cost = remain_cost
                strategy = (dp, pp, status)
        idx = len(future_intervals)
        while idx > 1:
            strategy = prev[idx][strategy]
            idx -= 1
        _, n_in, n_out = future_intervals[0]
        dp, pp, _ = strategy
        prob, cost = get_store().get((N, D, P, n_in, n_out))[get_store().encode_key((dp, pp))]
        return strategy, prob, cost
