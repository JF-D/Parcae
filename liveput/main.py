import os
import argparse
import multiprocessing
import json
import time

from cost import CostModel, setup_costmodel
from model import ModelSpec, gen_model_signature
from monte_carlo import LiveStore, setup_store, get_store
from trace import SpotTrace
from utils.graph import hist_graph, show_live_simualtion


parser = argparse.ArgumentParser()
parser.add_argument('--trace', type=str, default=None, help='trace file path')
parser.add_argument('--start-hour', type=float, default=0, help='trace start time')
parser.add_argument('--end-hour', type=float, default=None, help='trace end time')
parser.add_argument('--model', type=str, default='gpt-2')
parser.add_argument('--train-batch-size', type=int, default=256)
parser.add_argument('--look-ahead', type=int, default=10, help='look ahead for liveput')
parser.add_argument('--restart-cost', type=int, default=None, help='Restart cost, deafult is profile result')
parser.add_argument('--nnodes', type=int, default=32)
parser.add_argument('--ngpu-per-node', type=int, default=1)
parser.add_argument('--disable-pred', action='store_true')
# monte-carlo simulation test arguments
parser.add_argument('--N', type=int, default=8)
parser.add_argument('--D', type=int, default=2)
parser.add_argument('--P', type=int, default=4)
parser.add_argument('--n-in', type=int, default=0)
parser.add_argument('--n-out', type=int, default=2)
# monte-carlo simulation arguments
parser.add_argument('--n-sim', type=int, default=10000, help='monte carlo simulation times')
parser.add_argument('--update-cache', action='store_true',
                    help='update cache by re-run simulation')
parser.add_argument('--mc-sample', action='store_true',
                    help='run monte carlo simulation only')
parser.add_argument('--mc-sample-look-aheads', nargs='+', type=int,
                    help='run monte carlo simulation with the provided look ahead values')
# run liveput simulation for all cases
parser.add_argument('--liveput-simulation', action='store_true')
parser.add_argument('--liveput-simulation-look-aheads', nargs='+', type=int,
                    help='run liveput simulation with the provided look ahead values')
args = parser.parse_args()


def setup_env(model, restart_cost, nnodes=32, ngpu_per_node=1):
    model_signature = gen_model_signature(model, nnodes, ngpu_per_node)
    costmodel = CostModel(model=model.name, model_signature=model_signature, restart_cost=restart_cost)
    setup_costmodel(costmodel)

    store = LiveStore('livestore', model, restart_cost=restart_cost, n_sim=args.n_sim, update_cache=args.update_cache)
    setup_store(store)


def run_monte_carlo(args):
    N, D, P, n_in, n_out, update_cache = args
    key = (N, D, P, n_in, n_out)
    if update_cache or get_store().get(key) is None:
        get_store().simulate(N, D, P, n_in, n_out)
    values = get_store().get(key)
    return key, values


def monte_carlo_simulation(model, tracefile, start_hour, end_hour, interval_length, mc_sample_look_aheads, n_sim=1000, update_cache=False):
    trace_pairs = []

    trace = SpotTrace(tracefile, start_hour, end_hour, interval_length=interval_length, num_future_intervals=mc_sample_look_aheads[0])
    for i in range(trace.num_intervals):
        interval = trace.get_next_intervals(i, num_intervals=1, predict=False)[0]
        nnodes, n_in, n_out = interval
        prev_N = trace.get_node_number(i - 1)
        if (prev_N, n_in, n_out) not in trace_pairs:
            trace_pairs.append((prev_N, n_in, n_out))

    if int(os.environ.get('GPUS_PER_NODE', 1)) == 1:
        for _ in range(3):
            # run the prediction model for 15 times to simulate difference randomness
            for look_ahead in mc_sample_look_aheads:
                print(f'Try look ahead: {look_ahead}')
                trace = SpotTrace(tracefile, start_hour, end_hour, interval_length=interval_length, num_future_intervals=look_ahead)
                try:
                    for i in range(trace.num_intervals):
                        prev_N = trace.get_node_number(i - 1) # here, N is the prev_interval
                        future_intervals = trace.get_next_intervals(i, num_intervals=look_ahead, predict=True)
                        for fid, interval in enumerate(future_intervals):
                            nnodes, n_in, n_out = interval
                            if (prev_N, n_in, n_out) not in trace_pairs and n_out <= prev_N:
                                trace_pairs.append((prev_N, n_in, n_out))
                            prev_N = prev_N + n_in - n_out
                            assert prev_N == nnodes
                except Exception as e:
                    print(f'In look ahead {look_ahead}, find exception: {e}')
                    print('[WARNING] If you want to do monte carlo for prediction model, '
                          'please make sure the prediction model can pass the unit test.')
    else:
        print('We disable prediction for multi-gpu instances')

    def gen_strategy_pairs(nnodes):
        ret = []
        for dp in range(1, nnodes + 1):
            for pp in range(1, nnodes + 1):
                if dp * pp <= nnodes:
                    ret.append((dp, pp))
        return ret

    args = []
    for N, n_in, n_out in trace_pairs:
        for D, P in gen_strategy_pairs(N):
            args.append((N, D, P, n_in, n_out, update_cache))

    print(f'Begin to run monte carlo simulation, totoal tasks: {len(args)}, cpu count: {multiprocessing.cpu_count()}')
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        progress = 0
        st = time.time()
        for result in pool.imap_unordered(run_monte_carlo, args):
            key, value = result[0], result[1]
            try:
                get_store().update(key, value)
            except:
                print(key, value)

            progress += 1
            if progress % 1000 == 0:
                duration = time.time() - st
                print(f'> Saving store..., average speed: {progress / duration:.2f} tasks/s')
                get_store().save()
                duration = time.time() - st
                print(f'> Finish {progress}/{len(args)}, monte-carlo takes total {duration:.3f} seconds...')

    # save simulation result
    get_store().save()


def run_throughput_optimized(model, trace, ideal=False):
    from throughput import ThroughputOptimized
    opt = ThroughputOptimized(model, trace)
    hist = opt.optimize(ideal=ideal)
    return hist


def run_liveput_optimized(model, trace, look_ahead=1, predict=False):
    from liveput import LiveputOptimized
    opt = LiveputOptimized(model, trace)
    hist = opt.optimize(look_ahead=look_ahead, predict=predict)
    return hist


def run_liveput_optimization(tracefile, start_hour, end_hour, model, look_ahead, restart_cost, disable_pred, interval_length=60, update_cache=False):
    trace = SpotTrace(tracefile, start_hour, end_hour, interval_length=interval_length, num_future_intervals=look_ahead)

    hists, labels = [], []
    tpt_optimized_hist = run_throughput_optimized(model, trace, ideal=True)
    hists.append(tpt_optimized_hist)
    labels.append('Ideal')

    tpt_optimized_hist = run_throughput_optimized(model, trace, ideal=False)
    hists.append(tpt_optimized_hist)
    labels.append('Throughput Optimized')

    liveput_optimized_hist = run_liveput_optimized(model, trace, look_ahead=look_ahead, predict=False)
    hists.append(liveput_optimized_hist)
    labels.append('Liveput Optimized (Truth)')

    if not disable_pred:
        liveput_optimized_hist = run_liveput_optimized(model, trace, look_ahead=look_ahead, predict=True)
        hists.append(liveput_optimized_hist)
        labels.append('Liveput Optimized (Predict)')

    improvements, future_spaces = [], []
    out_name = f'log/test-{model.name}-rc{restart_cost}-ahead{look_ahead}.pdf' if restart_cost else f'log/test-{model.name}-ahead{look_ahead}.pdf'
    for idx, (hist, label) in enumerate(zip(hists, labels)):
        ideal_opt = hists[0][-1][1]
        tpt_opt = hists[1][-1][1]
        improvement = ((hist[-1][1] - tpt_opt) / tpt_opt) * 100
        future_space = ((ideal_opt - hist[-1][1]) / ideal_opt) * 100
        improvements.append(improvement)
        future_spaces.append(future_space)
        labels[idx] = f'{label} (vs. tpt: {improvement:.2f}%, vs. ideal: -{future_space:.2f}%)'
        print(f'{label} total samples: {hist[-1][1]:.3f}, improvement: {improvement:.2f}%, future space: {future_space:.2f}%')
    hist_graph(hists, labels, mark=('fake' in tracefile), out=out_name)

    # save simulation result
    # if update_cache or get_store().has_new_keys:
    #     get_store().save()

    key = f'{model.name}-rc{restart_cost}-ahead{look_ahead}'
    ret_value = []
    for improvement in improvements[2:]:
        ret_value.append(improvement)
    ret_value.append(future_spaces[2])
    return key, ret_value

def run_one_liveput_simulation(sim_args):
    trace_file, start_hour, end_hour, model_name, train_batch_size, look_ahead, restart_cost = sim_args
    disable_pred = False
    interval_length = 60
    update_cache = False

    model = ModelSpec.build(model=model_name, train_batch_size=train_batch_size, nparts=0)
    setup_env(model, restart_cost)
    try:
        ret = run_liveput_optimization(trace_file, start_hour, end_hour, model, look_ahead, restart_cost, disable_pred, interval_length=interval_length, update_cache=update_cache)
    except:
        print(f'Fail to simulate {sim_args}')
        ret = None
    return ret


def run_liveput_simulation(trace_file, start_hour, end_hour, liveput_simulation_look_aheads):
    models_to_bs = {
        'gpt-1.5b': 256,
        'gpt-2.7b': 256,
        'gpt-7.5b': 512,
        'resnet50': 2048,
    }
    all_restart_costs = [17, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300]

    all_args = []
    for model_name, train_batch_size in models_to_bs.items():
        for restart_cost in all_restart_costs:
            cachefile = f'livestore/{model_name}-rc{restart_cost}-bsz{train_batch_size}.json'
            if not os.path.exists(cachefile):
                continue
            for look_ahead in liveput_simulation_look_aheads:
                all_args.append((trace_file, start_hour, end_hour, model_name, train_batch_size, look_ahead, restart_cost))

    sim_res = {}
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        progress = 0
        st = time.time()
        for res in pool.imap_unordered(run_one_liveput_simulation, all_args):
            if res is not None:
                (key, ret) = res
                sim_res[key] = ret
            progress += 1
            duration = time.time() - st
            if progress % 5 == 0:
                print(f'> Finish {progress}/{len(all_args)}, takes total {duration:.3f} seconds...')

    show_live_simualtion(sim_res)
    trace_name = trace_file.split('/')[-1].split('.')[0]
    filename = f'log/livesim-{trace_name}.json'
    if os.path.exists(filename):
        with open(filename, 'r') as fp:
            load_res = json.load(fp)
        for key, value in load_res.items():
            if key not in sim_res:
                sim_res[key] = value
    with open(filename, 'w') as fp:
        json.dump(sim_res, fp)


def main():
    if args.trace is None:
        run_monte_carlo((args.N, args.D, args.P, args.n_in, args.n_out))
        return

    if args.liveput_simulation:
        run_liveput_simulation(args.trace, args.start_hour, args.end_hour, args.liveput_simulation_look_aheads)
        return

    model = ModelSpec.build(model=args.model, train_batch_size=args.train_batch_size, nparts=0)
    setup_env(model, args.restart_cost)

    if args.mc_sample:
        monte_carlo_simulation(model, args.trace, args.start_hour, args.end_hour, 60, args.mc_sample_look_aheads, args.n_sim, args.update_cache)
        return

    run_liveput_optimization(args.trace, args.start_hour, args.end_hour, model, args.look_ahead, args.restart_cost, args.disable_pred, update_cache=args.update_cache)

    # save simulation result
    if args.update_cache or get_store().has_new_keys:
        get_store().save()


if __name__ == '__main__':
    main()
