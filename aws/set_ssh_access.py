import os
import argparse
import subprocess

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', action='store_true',
                        help='Copy set_ssh_access.py to all instances.')
    parser.add_argument('--generate-keys-master', action='store_true',
                        help='Generate public keys on all instances.')
    parser.add_argument('--generate-keys', action='store_true',
                        help='Generate public keys on all instances.')
    parser.add_argument('--set-keys-master', action='store_true',
                        help='Set public keys for all instances.')
    parser.add_argument('--set-keys', action='store_true',
                        help='Set public keys for all instances.')
    args = parser.parse_args()
    return args


HOSTFILE = 'hostname'
KEYFILE = '/home/ubuntu/env/jiangfei.pem'
HOMEDIR = '/home/ubuntu'
tmpdir = '/tmp/tmp_ssh_keys'


def test_connection(ip_lists):
    processes = []
    for ip in ip_lists:
        cmd = f'ssh -i {KEYFILE} -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@{ip} "echo test > ~/test.txt"'
        p = subprocess.Popen(cmd, shell=True)
        processes.append(p)
    access_ip_lists = []
    for i, p in enumerate(processes):
        exit_code = p.wait()
        if exit_code == 0:
            access_ip_lists.append(ip_lists[i])
    return access_ip_lists


def get_ips(ip_lists=None):
    if ip_lists is not None:
        return ip_lists

    ip_lists = []
    with open(HOSTFILE, 'r') as f:
        for ip in f.readlines():
            ip = ip.strip()
            ip_lists.append(ip)
    return ip_lists


def init(ip_lists=None):
    access_ip_lists = test_connection(get_ips(ip_lists))

    processes = []
    for ip in access_ip_lists:
        cmd = f'scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i {KEYFILE} ./set_ssh_access.py ubuntu@{ip}:/tmp/'
        print(cmd)
        p = subprocess.Popen(cmd, shell=True)
        processes.append(p)

    for p in processes:
        p.wait()
    return access_ip_lists


def generate_keys():
    if not os.path.exists(f'{HOMEDIR}/.ssh/id_rsa.pub'):
        cmd = f'ssh-keygen -t rsa -N "" -f {HOMEDIR}/.ssh/id_rsa'
        p = subprocess.Popen(cmd, shell=True)
        p.wait()


def generate_keys_master(ip_lists=None):
    processes = []
    for ip in get_ips(ip_lists):
        cmd = f'ssh -o StrictHostKeyChecking=no -i {KEYFILE} ubuntu@{ip} "/opt/conda/envs/pytorch/bin/python /tmp/set_ssh_access.py --generate-keys"'
        print(cmd)
        p = subprocess.Popen(cmd, shell=True)
        processes.append(p)

    for p in processes:
        p.wait()


def set_keys():
    all_keys, exist_keys = {}, set()
    with open(f'{HOMEDIR}/authorized_keys', 'r') as f:
        for line in f.readlines():
            if line.split()[-1].startswith('ubuntu'):
                ip = line.split()[-1].split('@')[-1]
                all_keys[ip] = line

    with open(f'{HOMEDIR}/.ssh/authorized_keys', 'r') as f:
        for line in f.readlines():
            if line.split()[-1].startswith('ubuntu'):
                ip = line.split()[-1].split('@')[-1]
                exist_keys.add(ip)

    for ip, key in all_keys.items():
        if ip not in exist_keys:
            with open(f'{HOMEDIR}/.ssh/authorized_keys', 'a') as f:
                f.write(key)

    os.system(f'rm -rf {HOMEDIR}/authorized_keys')
    os.system(f'rm -rf /tmp/set_ssh_access.py')


def set_keys_master(ip_lists=None):
    os.system(f'mkdir -p {tmpdir}')

    # collect keys
    processes = []
    for ip in get_ips(ip_lists):
        cmd = f'scp -i {KEYFILE} -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@{ip}:{HOMEDIR}/.ssh/id_rsa.pub {tmpdir}/{ip}.pub'
        print(cmd)
        p = subprocess.Popen(cmd, shell=True)
        processes.append(p)

    for p in processes:
        p.wait()

    with open(f'{tmpdir}/authorized_keys', 'w') as fp:
        for ip in get_ips(ip_lists):
            with open(f'{tmpdir}/{ip}.pub', 'r') as infp:
                fp.write(infp.read())

    # distribute keys
    processes = []
    for ip in get_ips(ip_lists):
        cmd = f'scp -i {KEYFILE} -o StrictHostKeyChecking=no -o ConnectTimeout=10 {tmpdir}/authorized_keys ubuntu@{ip}:{HOMEDIR}/'
        print(cmd)
        p = subprocess.Popen(cmd, shell=True)
        processes.append(p)

    for p in processes:
        p.wait()

    # set keys
    processes = []
    for ip in get_ips(ip_lists):
        cmd = f'ssh -i {KEYFILE} -o StrictHostKeyChecking=no ubuntu@{ip} "/opt/conda/envs/pytorch/bin/python /tmp/set_ssh_access.py --set-keys"'
        print(cmd)
        p = subprocess.Popen(cmd, shell=True)
        processes.append(p)

    for p in processes:
        p.wait()

    os.system(f'rm -rf {tmpdir}')


def configure_aws(ip_lists=None):
    access_ip_lists = test_connection(get_ips(ip_lists))

    processes = []
    for ip in access_ip_lists:
        cmd = f'scp -r -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i {KEYFILE} /home/ubuntu/.aws ubuntu@{ip}:/home/ubuntu/'
        print(cmd)
        p = subprocess.Popen(cmd, shell=True)
        processes.append(p)

    for p in processes:
        p.wait()


if __name__ == '__main__':
    args = parse_arg()

    if args.init:
        init()
        configure_aws()

    if args.generate_keys_master:
        generate_keys_master()

    if args.generate_keys:
        generate_keys()

    if args.set_keys:
        set_keys()

    if args.set_keys_master:
        set_keys_master()
