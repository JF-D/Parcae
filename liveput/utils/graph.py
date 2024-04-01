import os
import argparse
import json
import math
import matplotlib.pyplot as plt


def show_comparison(hist_dict, out=None):
    def process_hist(hist):
        intervals = []
        for i, element in enumerate(hist):
            if i == 0:
                prev_t, prev_samples = 0, 0
                continue
            if element[2] == '':
                overhead = element[0] - prev_t
                continue
            intervals.append((element[1] - prev_samples, element[2], element[3], overhead))
            prev_t = element[0]
            prev_samples = element[1]
        return intervals

    liveput_predict_intervals = None
    for key, hist in hist_dict.items():
        if 'Throughput Optimized' in key:
            throughput_intervals = process_hist(hist)
        elif 'Liveput Optimized (Truth)' in key:
            liveput_truth_intervals = process_hist(hist)
        elif 'Liveput Optimized (Predict)' in key:
            liveput_predict_intervals = process_hist(hist)
    if liveput_predict_intervals is None:
        liveput_predict_intervals = liveput_truth_intervals

    print(f'Total intervals: {len(throughput_intervals)}')
    print(f'Interval | A (Throughput-Opt) | A Overhead | B (Liveput (Truth)) | B Overhead | Strategy (A-B) | Truth GAP (B-A) | C (Liveput (Predict)) | C Overhead | Strategy (B-C) | Truth-Pred GAP (B-C)')
    print(f'---------|--------------------|------------|---------------------|------------|----------------|-----------------|-----------------------|------------|----------------|---------------------')
    for i in range(len(throughput_intervals)):
        # Note: we have 4 hist [ideal, throughput optimized, liveput (truth), liveput (predict)]
        throughpt_opt = throughput_intervals[i]
        liveput_truth = liveput_truth_intervals[i]
        liveput_predict = liveput_predict_intervals[i]
        interval = throughpt_opt[2]

        if throughpt_opt[0] == 0:
            truth_gap = (liveput_truth[0] - throughpt_opt[0])
            tag_1 = ' '
        else:
            truth_gap = 100 * (liveput_truth[0] - throughpt_opt[0]) / throughpt_opt[0]
            tag_1 = '%'
        if liveput_predict[0] == 0:
            pred_gap = (liveput_truth[0] - liveput_predict[0])
            tag_2 = ' '
        else:
            pred_gap = 100 * (liveput_truth[0] - liveput_predict[0]) / liveput_predict[0]
            tag_2 = '%'


        def fmt_s(strategy):
            return f'{strategy[0]:2d}x{strategy[1]:2d}'

        def fmt_i(interval):
            return f'{interval[0]:2d}, {interval[1]:2d}, {interval[2]:2d}'

        if throughpt_opt[1][0] != liveput_truth[1][0] or throughpt_opt[1][1] != liveput_truth[1][1]:
            tag_ab = '+'
        else:
            tag_ab = ' '
        if liveput_predict[1][0] != liveput_truth[1][0] or liveput_predict[1][1] != liveput_truth[1][1]:
            tag_bc = '+'
        else:
            tag_bc = ' '

        print(f'{fmt_i(interval)}  | {fmt_s(throughpt_opt[1])}  | {throughpt_opt[3]/1000:7.3f} | {fmt_s(liveput_truth[1])}  | {liveput_truth[3]/1000:7.3f} | {tag_ab} '
              f'|  {truth_gap:10.2f}{tag_1} |  {fmt_s(liveput_predict[1])}  | {liveput_predict[3]/1000:7.3f} | {tag_bc} |  {pred_gap:10.2f}{tag_2}')

    def get_strategy(element):
        return (element[0], element[1])

    # save result into json file
    trace_events = []
    cur_N = 0
    for i in range(len(throughput_intervals)):
        prev_N, n_in, n_out = throughput_intervals[i][2]
        assert prev_N == cur_N
        cur_N = prev_N + n_in - n_out
        tpt_strategy = get_strategy(throughput_intervals[i][1])
        live_truth_strategy = get_strategy(liveput_truth_intervals[i][1])
        live_pred_strategy = get_strategy(liveput_predict_intervals[i][1])
        trace_event = {
            'interval': [prev_N, n_in, n_out],
            'duration': 60_000, # in ms
            'throughut-opt': tpt_strategy,
            'liveput-truth': live_truth_strategy,
            'liveput-predict': live_pred_strategy,
        }
        trace_events.append(trace_event)

    if out:
        with open(out, 'w') as fp:
            json.dump(trace_events, fp)

    # hourly statistics
    num_hours = math.ceil(len(throughput_intervals) / 60)
    print(f'Hourly statistics (total intervals: {len(throughput_intervals)})')
    print(f'Hour  | #event (+, -) | #avg_instances | Throughput-Opt  |  Liveput (Truth)  | Improvement | Liveput (Predict) | Improvement')
    print(f'------|---------------|----------------|-----------------|-------------------|-------------|-------------------|------------')
    for i in range(num_hours * 2):
        tpt_intervals = throughput_intervals[i * 30:(i + 2) * 30]
        live_truth_intervals = liveput_truth_intervals[i * 30:(i + 2) * 30]
        live_pred_intervals = liveput_predict_intervals[i * 30:(i + 2) * 30]
        if len(tpt_intervals) <= 0:
            continue

        avg_nodes = 0
        num_node_change, in_event, out_event = 0, 0, 0
        tpt_samples, live_truth_samples, live_pred_samples = 0, 0, 0
        tpt_num_st, live_truth_st, live_pred_st = 0, 0, 0

        prev_tpt_strategy = get_strategy(tpt_intervals[0][1])
        prev_live_truth_strategy = get_strategy(live_truth_intervals[0][1])
        prev_live_pred_strategy = get_strategy(live_pred_intervals[0][1])
        for j in range(len(tpt_intervals)):
            prev_N, n_in, n_out = tpt_intervals[j][2]
            if n_in + n_out > 0:
                num_node_change += 1
            if n_in > 0:
                in_event += 1
            if n_out > 0:
                out_event += 1
            avg_nodes += (prev_N + n_in - n_out) / 60
            tpt_samples += tpt_intervals[j][0]
            live_truth_samples += live_truth_intervals[j][0]
            live_pred_samples += live_pred_intervals[j][0]

            tpt_strategy = get_strategy(tpt_intervals[j][1])
            live_truth_strategy = get_strategy(live_truth_intervals[j][1])
            live_pred_strategy = get_strategy(live_pred_intervals[j][1])
            if prev_tpt_strategy != tpt_strategy:
                tpt_num_st += 1
            if prev_live_truth_strategy != live_truth_strategy:
                live_truth_st += 1
            if prev_live_pred_strategy != live_pred_strategy:
                live_pred_st += 1
            prev_tpt_strategy = tpt_strategy
            prev_live_truth_strategy = live_truth_strategy
            prev_live_pred_strategy = live_pred_strategy

        if tpt_samples == 0:
            truth_improv = (live_truth_samples - tpt_samples)
            tag_1 = ' '
            pred_improv = (live_pred_samples - tpt_samples)
            tag_2 = ' '
        else:
            truth_improv = 100 * (live_truth_samples - tpt_samples) / tpt_samples
            tag_1 = '%'
            pred_improv = 100 * (live_pred_samples - tpt_samples) / tpt_samples
            tag_2 = '%'
        print(f'{i/2:4.1f}  | {num_node_change:2d} ({in_event:2d} + {out_event:2d}) | {avg_nodes:5.2f} '
              f'| {tpt_samples:10.2f} ({tpt_num_st:2d}) '
              f'| {live_truth_samples:10.2f} ({live_truth_st:2d}) | {truth_improv:6.2f}{tag_1} '
              f'| {live_pred_samples:10.2f} ({live_pred_st:2d}) | {pred_improv:6.2f}{tag_2}')

def hist_graph(hists, labels=None, mark=False, out=None):
    fig, ax = plt.subplots(figsize=(6, 4))

    for idx, hist in enumerate(hists):
        x = [h[0] / 1000 for h in hist]
        y = [h[1] for h in hist]

        label = labels[idx] if labels else None
        ax.plot(x, y, label=label)

    # xticks, xticks_hour, h = [0], [0], 1
    # while True:
    #     xticks.append(h * 3600_000)
    #     xticks_hour.append(h)
    #     if h * 3600_000 > max_x:
    #         break
    #     h += 1
    # ax.set_xticks(xticks, xticks_hour)

    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Total Samples')
    ax.legend()
    if out:
        fig.savefig(out)

    # save result into json file
    hist_dict = {}
    for i, hist in enumerate(hists):
        label = i
        if labels is not None:
            label = labels[i]
        hist_dict[label] = hist

    if os.environ.get('OUT_FILENAME', out):
        out = os.environ.get('OUT_FILENAME', out)
    else:
        out = out[:-4] + '-trace'

    if out:
        filename = out + '-hist.json'
        with open(filename, 'w') as fp:
            json.dump(hist_dict, fp)

    print(f'Simulation result output: {out}.json')
    show_comparison(hist_dict, out=f'{out}.json')


def show_live_simualtion(sim_res):
    print(f'Model        | Restart Cost (s) | Look ahead (#) | Liveput (Truth) | Liveput (Predict) | Future Space')
    print('-------------|------------------|----------------|-----------------|-------------------|-------------')
    for model_name in ['gpt-1.5b', 'gpt-2.7b', 'gpt-7.5b', 'resnet50']:
        for restart_cost in [17, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300]:
            for look_ahead in [8, 10, 12]:
                key = f'{model_name}-rc{restart_cost}-ahead{look_ahead}'
                if key not in sim_res:
                    continue
                live_truth, live_pred, future_space = sim_res[key]
                print(f'{model_name}     | {restart_cost:16} | {look_ahead:14} | {live_truth:14.2f}% | {live_pred:16.2f}% | {future_space:8.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hist', type=str, default='log/test-gpt-1.5b-hist.json')
    parser.add_argument('--out-trace', type=str, default=None)
    args = parser.parse_args()

    with open(args.hist, 'r') as fp:
        hist_dict = json.load(fp)

    show_comparison(hist_dict, out=args.out_trace)
