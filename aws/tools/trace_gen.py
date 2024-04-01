import argparse
import dataclasses
import math
import json
import random


parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=32)
parser.add_argument('-trace', type=str, default=None)
parser.add_argument('-fix-interval', type=int, default=None)
parser.add_argument('-start-hour', type=float, default=0)
parser.add_argument('-end-hour', type=float, default=None)
parser.add_argument('-output-trace', type=str, default='trace.txt')
parser.add_argument('-skip-no-op', action='store_true')
args = parser.parse_args()

HOSTS = 'hostname'
TRACE = args.output_trace


def get_hosts():
    hosts = {}
    with open(HOSTS, 'r') as f:
        for i, line in enumerate(f.readlines()):
            hosts[i] = line.strip()
            if len(hosts) == args.n:
                break
    return hosts


@dataclasses.dataclass
class TraceInterval:
    interval: list
    duration: float
    throughput_opt_strategy: tuple
    liveput_truth_strategy: tuple
    liveput_predict_strategy: tuple


class TraceIntevals:
    def __init__(self, trace_file, fix_interval=None, start_hour=0, end_hour=None, skip_no_op=False):
        self.trace_file = trace_file
        self.fix_interval = fix_interval
        self.skip_no_op = skip_no_op

        start_timestamp = int(start_hour * 3600_000)
        if end_hour is not None:
            end_timestamp = int(end_hour * 3600_000)
        else:
            end_timestamp = None
        self.read(trace_file, start_timestamp=start_timestamp, end_timestamp=end_timestamp)

    def read(self, trace_file, start_timestamp=0, end_timestamp=None):
        self.intervals = []
        self.interval_timestamps = []
        throughput_opts = {}

        if args.trace.endswith('.json'):
            print('Using version 2 trace: preprocessed by liveput optimization')
            with open(args.trace, 'r') as fp:
                trace_intervals = json.load(fp)
            timestamp = 0
            for trace_interval in trace_intervals:
                interval = trace_interval['interval']
                duration = int(trace_interval['duration'])
                throughput_opt_st = trace_interval.get('throughut-opt', None)
                liveput_truth_st = trace_interval.get('liveput-truth', None)
                liveput_predict_st = trace_interval.get('liveput-predict', None)

                N = interval[0] + interval[1] - interval[2]
                throughput_opts[N] = throughput_opt_st

                if timestamp < start_timestamp:
                    timestamp += duration
                    continue

                if end_timestamp is not None and timestamp >= end_timestamp:
                    break

                self.intervals.append(TraceInterval(
                    interval, duration, throughput_opt_st, liveput_truth_st, liveput_predict_st
                ))
                self.interval_timestamps.append(timestamp - start_timestamp)
                timestamp += duration
        else:
            print('Currently not support this format of trace file')
            exit()

        if start_timestamp > 0:
            # modify the initial event
            prev_N, n_in, n_out = self.intervals[0].interval
            N_0 = prev_N + n_in - n_out
            self.intervals[0].interval = [0, N_0, 0]

        if self.skip_no_op:
            # self.intervals = [interval for interval in self.intervals if interval.interval[1] > 0 or interval.interval[2] > 0]
            # self.interval_timestamps = [i * 90_000 for i in range(len(self.intervals))]

            new_intervals = []
            prev_interval = None
            for interval in self.intervals:
                # can fuse to previous interval
                n_add = interval.interval[1]
                n_remove = interval.interval[2]
                if prev_interval is not None:
                    pred_no_change = tuple(prev_interval.liveput_predict_strategy) == tuple(interval.liveput_predict_strategy)
                    truth_no_change = tuple(prev_interval.liveput_truth_strategy) == tuple(interval.liveput_truth_strategy)
                else:
                    pred_no_change = False
                    truth_no_change = False
                if prev_interval is not None and n_add == 0 and n_remove == 0 and pred_no_change and truth_no_change:
                    new_intervals[-1].duration += interval.duration
                else:
                    new_intervals.append(interval)
                prev_interval = interval

            self.interval_timestamps = []
            tstamp = 0
            for interval in new_intervals:
                if self.fix_interval * 1000 < interval.duration:
                    interval.duration = self.fix_interval * 1000
                self.interval_timestamps.append(tstamp)
                tstamp += interval.duration
            self.intervals = new_intervals

        for n in sorted(throughput_opts.keys()):
            print(f'{n}: {throughput_opts[n]},')

    def gen_replay_trace(self, output_file):
        replay_trace = []

        fp = open(output_file, 'w')
        self.node_id = 0
        cur_nodes = []
        for i, interval in enumerate(self.intervals):
            timestamp = self.interval_timestamps[i]
            prev_N, n_in, n_out = interval.interval
            throughput_opt_st = interval.throughput_opt_strategy
            liveput_truth_st = interval.liveput_truth_strategy
            liveput_predict_st = interval.liveput_predict_strategy

            N = prev_N + n_in - n_out
            nodes = []
            if n_in > 0:
                operation = 'add'

                # new hosts
                for i in range(n_in):
                    new_node = f'node-{self.node_id}'
                    nodes.append(new_node)
                    cur_nodes.append(new_node)
                    self.node_id += 1
            elif n_out > 0:
                # remove hosts
                remove_nodes = random.sample(cur_nodes[1:], n_out)
                operation = 'remove'

                for node_id in remove_nodes:
                    cur_nodes.remove(node_id)
                    nodes.append(node_id)
            else:
                operation = 'no-op'
                if prev_liveput_predict_st[0] != liveput_predict_st[0] or prev_liveput_predict_st[1] != liveput_predict_st[1]:
                    print(f'Warning: liveput prediction changed at {timestamp}: {prev_liveput_predict_st} -> {liveput_predict_st}')
                    operation = 'migration'
                elif liveput_truth_st[0] != prev_liveput_truth_st[0] or liveput_truth_st[1] != prev_liveput_truth_st[1]:
                    print(f'Warning: liveput truth changed at {timestamp}: {prev_liveput_predict_st} -> {liveput_predict_st}')
                    operation = 'migration'

            prev_liveput_truth_st = liveput_truth_st
            prev_liveput_predict_st = liveput_predict_st

            assert len(nodes) == n_in + n_out
            assert len(cur_nodes) == N
            event = {
                'nodes': nodes,
                'duration': interval.duration,
                'throughput_opt_strategy': throughput_opt_st,
                'liveput_truth_strategy': liveput_truth_st,
                'liveput_predict_strategy': liveput_predict_st,
            }
            event = [timestamp, operation, event]
            fp.write(f'{json.dumps(event)}\n')

        fp.close()
        return replay_trace

    def preemption_rate(self):
        # self.intervals = self.intervals[30:]
        num_hours = math.ceil(len(self.intervals) / 60)
        # print(f'Hour | Avg Preemption Rate | #node preempted | Avg node | Preemption Rates')
        # print(f'-----|--------------------|-----------------|-----------|-----------------')
        print(f'Hour | A1 | A2 | #preemptions | #additions | #Avg node')
        print(f'-----|----|----|--------------|------------|----------')
        for i in range(num_hours):
            for offset in [0, 0.5]:
                hour_intervals = self.intervals[int((i + offset) * 60): int((i + offset + 1) * 60)]
                preemption_rates = []
                addition_rates = []
                total_new_nodes, total_preempted_nodes = 0, 0
                avg_nodes = 0
                total_new_nodes = hour_intervals[0].interval[0]
                for interval in hour_intervals:
                    prev_N, n_in, n_out = interval.interval
                    total_preempted_nodes += n_out
                    total_new_nodes += n_in
                    if n_out != 0:
                        preemption_rates.append(n_out / 32)
                    if n_in != 0:
                        addition_rates.append(n_in / 32)
                    avg_nodes += (prev_N + n_in - n_out) /60
                a1_avg_rate = 100 * sum(preemption_rates) / len(preemption_rates) if len(preemption_rates) > 0 else 0.0
                a2_avg_rate = 100 * total_preempted_nodes / total_new_nodes
                # print(f'{i+0.5:2.1f} | {a1_avg_rate:4.1f}% | {a2_avg_rate:4.1f}% | {len(preemption_rates):2d} | {len(addition_rates):2d} | {avg_nodes:4.1f} ')
                print(f'{i+offset:2.1f} | {a1_avg_rate:4.1f}% | {a2_avg_rate:4.1f}% | {len(preemption_rates):2d} | {len(addition_rates):2d} | {avg_nodes:4.1f} ')

if __name__ == '__main__':
    # hosts = get_hosts()
    trace_generator = TraceIntevals(args.trace, args.fix_interval, start_hour=args.start_hour, end_hour=args.end_hour, skip_no_op=args.skip_no_op)
    trace_generator.gen_replay_trace(TRACE)
    # trace_generator.gen_replay_trace(hosts, TRACE)
    trace_generator.preemption_rate()
