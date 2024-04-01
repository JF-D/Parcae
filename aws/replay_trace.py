import os
import sys
import argparse
import logging
import json
import uuid
import time
import socket
import subprocess

from torch.distributed.argparse_util import env
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.utils import _parse_rendezvous_config
from torch.distributed.run import parse_min_max_nnodes

from project_pactum.rendezvous.etcd import create_rdzv_handler


# constants
GRACE_PERIOD = 25_000 # ms
WORKDIR = 'Parcae/examples/megatron/Megatron-LM-v1.1.5-3D_parallelism'
EXECUTABLE = 'bash'
SCRIPTS = 'examples/spotdl.sh'
FAILURE_DELAY_STEPS = 2

parser = argparse.ArgumentParser(description='Replay a trace file')
parser.add_argument('--trace', type=str, required=True, help='Trace file to replay')
parser.add_argument('--n', type=int, default=32, help='Number of nodes')
parser.add_argument('--hostfile', type=str, default='hostname', help='Hostfile')
parser.add_argument('--gpu_per_node', type=int, default=1)
parser.add_argument('--dry-run', action='store_true', help='Dry run the trace')
parser.add_argument('--replayer-log', type=str, default='log/replayer.log', help='The logfile of replayer')
parser.add_argument('--approach', type=str, default='liveput-truth', help='Test approach')
parser.add_argument('--rdzv-ip', type=str, default='172.31.28.108', help='Rendezvous IP')
parser.add_argument('--rdzv-port', type=str, default='2134', help='Rendezvous Port')
parser.add_argument('--nccl-debug', action='store_true', help='Enable NCCL Debug')
parser.add_argument(
    "--nnodes",
    action=env,
    type=str,
    default="1:1",
    help="Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>.",
)
parser.add_argument(
    "--rdzv_backend",
    action=env,
    type=str,
    default="static",
    help="Rendezvous backend.",
)
# parser.add_argument(
#     "--rdzv_endpoint",
#     action=env,
#     type=str,
#     default="",
#     help="Rendezvous backend endpoint; usually in form <host>:<port>.",
# )
parser.add_argument(
    "--rdzv_id",
    action=env,
    type=str,
    default="none",
    help="User-defined group id.",
)
parser.add_argument(
    "--rdzv_conf",
    action=env,
    type=str,
    default="",
    help="Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).",
)
args = parser.parse_args()
args.rdzv_endpoint = f'{args.rdzv_ip}:{args.rdzv_port}'


logging.basicConfig(filename=args.replayer_log,
                    filemode='w',
                    format='[%(asctime)s] %(message)s',
                    level=logging.INFO,)
logger = logging.getLogger()
# sys.stderr.write = logger.error


class TraceReplayer:
    def __init__(self, trace_file, n=32, hostfile='hostname', gpu_per_node=1, dry_run=False):
        self.trace_file = trace_file
        self.n = n
        self.hostfile = hostfile
        self.dry_run = dry_run
        self.gpu_per_node = gpu_per_node
        self.read_trace(trace_file)

        self.rdzv_id = str(uuid.uuid4())
        self.replayer_ip = socket.gethostbyname(socket.gethostname())
        self.is_store_connected = False
        self.global_steps = 0

        self.node_remove_time = {}
        self.profiler = False

        os.system('rm -rf log/ssh_out_logs/*')
        os.system('rm -rf log/ssh_err_logs/*')

    def _setup_rdzv(self):
        min_nodes, max_nodes = parse_min_max_nnodes(args.nnodes)
        assert 0 < min_nodes <= max_nodes
        self.rdzv_configs = _parse_rendezvous_config(args.rdzv_conf)

        if "last_call_timeout" not in self.rdzv_configs:
            self.rdzv_configs["last_call_timeout"] = 5
        if "timeout" not in self.rdzv_configs:
            self.rdzv_configs["timeout"] = 900

        self.rdzv_configs["last_call_timeout"] = int(self.rdzv_configs["last_call_timeout"])
        self.rdzv_configs["timeout"] = int(self.rdzv_configs["timeout"])

        rdzv_parameters = RendezvousParameters(
            backend=args.rdzv_backend,
            endpoint=args.rdzv_endpoint,
            run_id=self.rdzv_id,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            **self.rdzv_configs,
        )

        self.rdzv_handler = create_rdzv_handler(rdzv_parameters, global_scheduler=False)
        self.fail_lock = self.rdzv_handler.create_lock('fail-lock')

    def setup_store(self):
        try:
            self.global_store = self.rdzv_handler.setup_kv_store()
            self.is_store_connected = True
        except:
            self.is_store_connected = False
            pass

    def read_trace(self, trace_file):
        self.hosts = {}
        with open(self.hostfile, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if self.dry_run:
                    ip = line.strip()
                else:
                    ip = socket.gethostbyname(line.strip())
                self.hosts[i] = ip
                if len(self.hosts) == self.n:
                    break

        # old read trace
        self.trace = []
        new_version_trace = False
        with open(trace_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('#'):
                    continue
                event = json.loads(line)
                if len(event[2]['nodes']) > 0 and event[2]['nodes'][0].startswith('node'):
                    new_version_trace = True
                    break

            if not new_version_trace:
                for line in lines:
                    # self-defined comment for trace file
                    if line.startswith('#'):
                        continue
                    event = json.loads(line)
                    self.trace.append(event)

        # new read trace
        if new_version_trace:
            self.trace = []
            remain_nodes = list(self.hosts.keys())
            node_map = {}
            with open(trace_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # self-defined comment for trace file
                    if line.startswith('#'):
                        continue
                    event = json.loads(line)
                    nodes = []
                    if event[1] == 'add':
                        for node_id in event[2]['nodes']:
                            host_id = remain_nodes.pop(0)
                            node_map[node_id] = host_id
                            nodes.append(self.hosts[host_id])
                    else:
                        for node_id in event[2]['nodes']:
                            host_id = node_map[node_id]
                            nodes.append(self.hosts[host_id])
                            remain_nodes.append(host_id)
                    event[2]['nodes'] = nodes
                    self.trace.append(event)

    def timer(self, init=False):
        if init:
            self.start_time = time.time()
        cur_time_stamp = (time.time() - self.start_time) * 1000
        return cur_time_stamp

    def is_self(self, ip):
        return self.replayer_ip == ip

    def setup_etcd(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        cur_env = os.environ.copy()
        cur_env['RDZV_IP'] = args.rdzv_ip
        cur_env['RDZV_PORT'] = args.rdzv_port

        launch_cmd = f'sh {cur_dir}/deploy_etcd.sh'

        if self.is_self(args.rdzv_ip):
            cmd = launch_cmd.split()
        else:
            cmd = ['ssh', f'ubuntu@{args.rdzv_ip}']
            cmd.append(f'RDZV_IP={args.rdzv_ip}')
            cmd.append(f'RDZV_PORT={args.rdzv_port}')
            cmd.append(f'sh {cur_dir}/deploy_etcd.sh')

        logger.info(f'Setup etcd server, CMD: {" ".join(cmd)}')
        subprocess.Popen(cmd, env=cur_env)

        # wait for etcd server to start
        time.sleep(1)

        self._setup_rdzv()

    def stop_node(self, ip):
        cmd = 'aws ec2 describe-instances --filter Name=private-ip-address,Values=172.31.3.95 --query "Reservations[].Instances[].InstanceId" --output text'
        instance_id = os.popen(cmd).read()
        instance_id = instance_id.split()[0]
        cmd = f'aws ec2 stop-instances --instance-ids {instance_id}'
        # os.popen(cmd)
        logger.info(f'Stop node {ip}, CMD: {cmd}')

    def stop_nodes(self):
        remain_nodes = {}
        for ip in self.node_remove_time:
            logger.info(f'Node {ip} removed at {self.node_remove_time[ip]}, current time: {self.timer()}')
            if self.timer() - self.node_remove_time[ip] > 60_000:
                self.stop_node(ip)
            else:
                remain_nodes[ip] = self.node_remove_time[ip]
        self.node_remove_time = remain_nodes

    def start_node(self, ip):
        cur_env = os.environ.copy()
        cur_env['RDZV_IP'] = args.rdzv_ip
        cur_env['RDZV_PORT'] = args.rdzv_port
        cur_env['RDZV_ID'] = self.rdzv_id
        cur_env['RDZV_CONF'] = args.rdzv_conf
        if args.nccl_debug:
            cur_env['NCCL_DEBUG'] = 'INFO'

        launch_cmd = f'{EXECUTABLE} {WORKDIR}/{SCRIPTS}'

        if self.is_self(ip):
            cmd = launch_cmd.split()
        else:
            cmd = ['ssh', f'ubuntu@{ip}']
            cmd.append(f'RDZV_IP={args.rdzv_ip}')
            cmd.append(f'RDZV_PORT={args.rdzv_port}')
            cmd.append(f'RDZV_ID={self.rdzv_id}')
            cmd.append(f'RDZV_CONF={args.rdzv_conf}')
            if args.nccl_debug:
                cmd.append('NCCL_DEBUG=INFO')
            cmd.append(launch_cmd)

        logger.info(f'>>> [{self.timer()/1000:.3f}] Start node {ip}, CMD: {" ".join(cmd)}')
        if not self.dry_run:
            # out_file = open(f"log/ssh_out_logs/{ip}.log", "a")
            # err_file = open(f"log/ssh_err_logs/{ip}.log", "a")
            # subprocess.Popen(cmd, env=cur_env, stdout=out_file, stderr=err_file)
            subprocess.Popen(cmd, env=cur_env)

    def remove_node(self, ip):
        cmd = ['ssh', f'ubuntu@{ip}']
        cmd.append('pkill -15 -f project_pactum.run')

        logger.info(f'>>> [{self.timer()/1000:.3f}] Remove node {ip}, CMD: {" ".join(cmd)}')
        if not self.dry_run:
            subprocess.Popen(cmd)

        # record remove time
        if self.profiler:
            self.node_remove_time[ip] = self.timer()

    def update_rank_map(self):
        try:
            _, state = self.rdzv_handler._rdzv_impl.get_rdzv_state()
        except:
            state = None
        logger.info(f'state: {state}')
        if state is None or state['status'] != 'final':
            return

        version = state['version']
        group_world_size = len(state['participants'])
        # num_pipeliens = int(state['num_pipelines'])
        # num_stages = int(state['num_stages'])
        # world_size = num_pipeliens * num_stages

        alive_members = self.rdzv_handler._rdzv_impl.client.get(self.rdzv_handler._rdzv_impl.get_path(f'/rdzv/v_{version}'))
        keep_alive_keys = [ch.key for ch in alive_members.children]
        self.rank_map = {}
        for key in state["keep_alives"]:
            if key.endswith("_coordinates"):
                continue

            if key not in keep_alive_keys:
                continue

            rank = self.rdzv_handler._rdzv_impl.rank_pattern.match(key).group(2)
            this_ip_key = self.rdzv_handler._rdzv_impl.get_path(
                "/rdzv/v_{}/rank_{}_ip".format(version, rank)
            )
            ip = self.rdzv_handler._rdzv_impl.client.get(this_ip_key).value

            coordinates_key = self.rdzv_handler._rdzv_impl.get_path(f'rdzv/v_{version}/rank_{rank}_coordinates')
            coordinates = json.loads(self.rdzv_handler._rdzv_impl.client.get(coordinates_key).value)
            if ip in self.cur_node_ips and len(coordinates) > 0:
            # if ip in self.cur_node_ips:
                self.rank_map[ip] = rank
                logger.info(f'>>> [{self.timer()/1000:.3f}]          Update rank map: {ip} -> {rank}')
            else:
                logger.info(f'>>> [{self.timer()/1000:.3f}] Find alive node not in current node list, ip: {ip}, rank: {rank}')

        self.global_steps = max(self.global_steps, int(self.global_store.get('global-steps')))

    def clean_failures(self):
        self.fail_lock.acquire()
        failures = json.loads(self.global_store.get('failures'))
        already_deleted = []
        for rank, step in failures.items():
            if step < self.global_steps:
                already_deleted.append(rank)

        for rank in already_deleted:
            del failures[rank]

        self.global_store.set('failures', json.dumps(failures))
        self.fail_lock.release()

    def issue_failures(self, fail_ips, strategy):
        self.fail_lock.acquire()
        failures = json.loads(self.global_store.get('failures'))
        already_deleted = []
        for rank, step in failures.items():
            if step < self.global_steps:
                already_deleted.append(rank)

        for rank in already_deleted:
            del failures[rank]

        global_step = self.global_steps + FAILURE_DELAY_STEPS
        for ip in fail_ips:
            if ip in self.rank_map:
                failures[str(self.rank_map[ip])] = global_step
            else:
                self.remove_node(ip)

        self.global_store.set('failures', json.dumps(failures))

        self.issue_strategy_plan(len(fail_ips), strategy)
        self.fail_lock.release()
        logger.info(f'>>> [{self.timer()/1000:.3f}] Issue failures: {failures}, strategy: {strategy} at step {self.global_steps}')

    def issue_strategy_plan(self, num_nodes, strategy):
        # strategy = (strategy[0] * self.gpu_per_node, strategy[1])
        strategy_key = self.rdzv_handler._rdzv_impl.get_path("/rdzv/next_strategy")
        strategy_plan = json.loads(self.rdzv_handler._rdzv_impl.client.get(strategy_key).value)

        already_deleted = []
        for step, _ in strategy_plan.items():
            if int(step) < self.global_steps:
                already_deleted.append(step)
        for step in already_deleted:
            del strategy_plan[step]

        global_step = self.global_steps + FAILURE_DELAY_STEPS
        strategy_plan[str(global_step)] = (num_nodes, strategy)
        self.rdzv_handler._rdzv_impl.client.write(strategy_key, json.dumps(strategy_plan))

        logger.info(f'>>> [{self.timer()/1000:.3f}] Issue strategy: {strategy} at step {self.global_steps}')
        logger.info(f'                     strategy plan: {strategy_plan}')

    def issue_init_strategy(self, strategy):
        # strategy = (strategy[0] * self.gpu_per_node, strategy[1])
        next_strategy_key = self.rdzv_handler._rdzv_impl.get_path("/rdzv/next_strategy")
        strategy_plan = {str(0): (0, strategy)}
        self.rdzv_handler._rdzv_impl.client.write(next_strategy_key, json.dumps(strategy_plan))
        logger.info(f'>>> [{self.timer()/1000:.3f}] set init strategy: {strategy}')

    def sleep(self, sec):
        if not args.dry_run:
            time.sleep(sec)

    def replay(self, approach='liveput-truth'):
        # preprocess trace
        self.approach = approach
        events = {}
        prev_timestamp = -1
        while len(self.trace) > 0:
            time_stamp, operation, event_info = self.trace.pop(0)
            if operation == 'remove':
                time_stamp = time_stamp - GRACE_PERIOD
            assert time_stamp not in events
            duration = event_info['duration']
            if approach == 'liveput-truth':
                strategy_name = 'liveput_truth_strategy'
            elif approach == 'liveput-predict':
                strategy_name = 'liveput_predict_strategy'
            strategy = event_info[strategy_name]
            events[time_stamp] = (operation, duration, event_info['nodes'], strategy)
            # make sure keep time order
            assert time_stamp > prev_timestamp
            prev_timestamp = time_stamp

        time_order_events = sorted(events.items(), key=lambda x: x[0])
        logger.info(f'Begin to replay trace {self.trace_file}')

        self.cur_node_ips = set()
        self.rank_map = {}
        prev_strategy = None
        self.timer(init=True)
        final_event = time_order_events[-1]
        while len(time_order_events) > 0:
            time_stamp, next_event = time_order_events.pop(0)
            operation = next_event[0]
            logger.info(f'>>> [{self.timer()/1000:.3f}] next event at {time_stamp}: {next_event}')

            # stop nodes
            if self.profiler:
                self.stop_nodes()

            if not self.dry_run:
                # always connect to latest store
                self.setup_store()
                while not self.is_store_connected:
                    cur_time_stamp = self.timer()
                    if cur_time_stamp + 1_000 < time_stamp:
                        self.sleep(1)
                    else:
                        break
                    logger.info(f'>>> [{self.timer()/1000:.3f}] Retry to connect to store')
                    self.setup_store()

            if prev_strategy is None and not self.dry_run:
                init_strategy = next_event[3]
                self.issue_init_strategy(init_strategy)

            if operation == 'no-op' and next_event[3] == prev_strategy:
                if not self.dry_run:
                    self.sleep(10) # sleep 10 seconds
                    self.update_rank_map()
                    self.clean_failures()

                cur_time_stamp = self.timer()
                if cur_time_stamp < time_stamp:
                    self.sleep((time_stamp - cur_time_stamp) / 1000)
                continue

            cur_time_stamp = self.timer()
            if cur_time_stamp + 300 < time_stamp:
                self.sleep((time_stamp - cur_time_stamp - 300) / 1000)

            if not self.dry_run:
                # # always connect to latest store
                self.setup_store()
                # while not self.is_store_connected:
                #     cur_time_stamp = self.timer()
                #     if cur_time_stamp + 1_000 < time_stamp:
                #         self.sleep(1)
                #     else:
                #         break
                #     logger.info(f'>>> [{self.timer()/1000:.3f}] Retry to connect to store')
                #     self.setup_store()
                if self.is_store_connected:
                    self.update_rank_map()
                    self.clean_failures()

                    logger.info(f'>>> [{self.timer()/1000:.3f}] infos:')
                    logger.info(f'             num: {len(self.cur_node_ips)}, {self.cur_node_ips}')
                    logger.info(f'             rank: {self.rank_map}')

            strategy = next_event[3]
            if not self.dry_run:
                # setup failures
                if operation == 'remove':
                    fail_ips = set()
                    for ip in next_event[2]:
                        fail_ips.add(ip)
                    if len(fail_ips) > 0:
                        self.issue_failures(fail_ips, strategy)
                    time_stamp = time_stamp + GRACE_PERIOD

                    # fail ips
                    remain_fail_ips = set()
                    for ip in fail_ips:
                        if ip not in self.rank_map:
                            # self.remove_node(ip)
                            pass
                        else:
                            remain_fail_ips.add(ip)
                    fail_ips = remain_fail_ips
                else:
                    num_new_nodes = len(next_event[2])
                    if prev_strategy is not None:
                        self.issue_strategy_plan(num_new_nodes, strategy)

            cur_time_stamp = self.timer()
            if cur_time_stamp < time_stamp:
                self.sleep((time_stamp - cur_time_stamp) / 1000)

            if operation == 'add':
                for ip in next_event[2]:
                    self.cur_node_ips.add(ip)
                    self.start_node(ip)
            elif operation == 'remove':
                # for ip in next_event[2]:
                for ip in fail_ips:
                    ip_addr = ip
                    self.cur_node_ips.remove(ip_addr)
                    if ip_addr in self.rank_map:
                        del self.rank_map[ip_addr]
                    self.remove_node(ip)

            prev_strategy = strategy

        # stop nodes
        if self.profiler:
            self.stop_nodes()

        # remove all nodes
        final_timestamp = final_event[0] + final_event[1][1]
        if final_event[1][0] == 'remove':
            final_timestamp += GRACE_PERIOD

        self.sleep(25)
        if not self.dry_run:
            self.setup_store()
            self.update_rank_map()

        cur_time_stamp = self.timer()
        if cur_time_stamp < final_timestamp - GRACE_PERIOD:
            self.sleep((final_timestamp - GRACE_PERIOD - cur_time_stamp) / 1000)
        fail_ips = set()
        for ip in self.cur_node_ips:
            fail_ips.add(ip)

        if len(fail_ips) > 0:
            if not self.dry_run:
                self.global_steps = max(self.global_steps, int(self.global_store.get('global-steps')))
                self.issue_failures(fail_ips, strategy)

        cur_time_stamp = self.timer()
        if cur_time_stamp < final_timestamp:
            self.sleep((final_timestamp - cur_time_stamp) / 1000)
        for ip in self.cur_node_ips:
            self.remove_node(ip)
        logger.info(f'>>> [{self.timer()/1000:.3f}] Replay trace {self.trace_file} finished')

        # kill etcd server
        self.sleep(2)
        os.popen('pkill -f etcd')

    def run(self, approach='liveput-truth'):
        if not self.dry_run:
            self.setup_etcd()
        self.replay(approach)

if __name__ == '__main__':
    replayer = TraceReplayer(args.trace, n=args.n, hostfile=args.hostfile, gpu_per_node=args.gpu_per_node, dry_run=args.dry_run)
    replayer.run(args.approach)
