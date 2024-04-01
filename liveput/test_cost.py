import os
import argparse
import multiprocessing
import json
import time

from cost import CostModel, setup_costmodel
from model import ModelSpec, gen_model_signature
from monte_carlo import LiveStore, setup_store, get_store, MonteCarloLiveput
from trace import SpotTrace
from utils.graph import hist_graph, show_live_simualtion


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt-1.5b')
parser.add_argument('--train-batch-size', type=int, default=128)
parser.add_argument('--overhead-log', type=str, default='log/overhead.log')
args = parser.parse_args()


def setup_env(model, restart_cost, nnodes=32, ngpu_per_node=1):
    model_signature = gen_model_signature(model, nnodes, ngpu_per_node)
    costmodel = CostModel(model=model.name, model_signature=model_signature, restart_cost=restart_cost)
    setup_costmodel(costmodel)

    # store = LiveStore('livestore', model, restart_cost=restart_cost)
    # setup_store(store)


def main():
    model = ModelSpec.build(model=args.model, train_batch_size=args.train_batch_size, nparts=0)
    setup_env(model, None)

    with open(args.overhead_log, 'r') as f:
        all_log_lines = f.readlines()
        for i, line in enumerate(all_log_lines):
            if line.startswith('Profile-Overhead:'):
                overhead = float(line.split()[1][:-2])
                prev_D = int(line.split()[2][1:-1])
                prev_P = int(line.split()[3][:-1])
                next_D = int(line.split()[5][1:-1])
                next_P = int(line.split()[6][:-1])

                if 'Intra-stage migration' in all_log_lines[i + 1]:
                    N = prev_D * prev_P
                    sampler = MonteCarloLiveput(model, N, prev_D, prev_P, 0, 0)
                    cost = sampler.estimate_intra_stage_migration(next_D, next_P)
                else:
                    nline = all_log_lines[i + 1]
                    n_in = int(nline.split()[-1])
                    preempted_nodes = eval(nline.split('New-nodes:')[0].split('Preempted-nodes:')[-1])
                    n_out = len(preempted_nodes)

                    if n_out > 0:
                        N = prev_D * prev_P + n_in
                        n_in = 0
                    else:
                        N = prev_D * prev_P
                    sampler = MonteCarloLiveput(model, N, prev_D, prev_P, n_in, n_out)
                    cost = sampler.try_migration(preempted_nodes, next_D, next_P)

                err = 100 * (cost - overhead * 1000) / (overhead * 1000)
                print(f'({prev_D:2d}x{prev_P:2d})->({next_D:2d}x{next_P:2d}): profile: {overhead*1000:9.2f}, estimation: {cost:9.2f}, err: {err:.2f}%')


if __name__ == '__main__':
    main()
