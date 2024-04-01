import os
import bisect
import json
import math
import numpy as np


MB = 1024 * 1024
INF = np.inf
DEFAULT_STRATEGY_CACHE_PATH = 'livestore/strategy_cache.json'

COSTMODEL = None
GPUS_PER_NODE = int(os.environ.get('GPUS_PER_NODE', 1))


def setup_costmodel(costmodel):
    global COSTMODEL
    COSTMODEL = costmodel


def get_costmodel():
    global COSTMODEL
    return COSTMODEL


class CostModel:
    """Cost model.

    This model estimates the cost of communication, migration and etc.
    We also use this model to estimate the cost of different strategies.
    """
    def __init__(self, model='gpt-2', model_signature=None, restart_cost=None):
        self.model = model
        self.model_signature = model_signature
        self.restart_cost = restart_cost

        self.best_strategy_of_nnodes = {}
        self.load_profile_result()
        self.load_dataset_cost = 0.0 # ms, this varies across datasets

    def load_profile_result(self):
        # bw = 10Gb/s, profiling result is in us
        self.p2p_volume_to_cost = {
                     4:  142.907,
                     8:  139.679,
                    16:  143.102,
                    32:  139.210,
                    64:  143.954,
                   128:  140.836,
                   256:  141.442,
                   512:  144.788,
                  1024:  137.372,
                  2048:  140.417,
                  4096:  141.651,
                  8192:  145.364,
                 16384:  154.935,
                 32768:  162.665,
                 65536:  194.492,
                131072:  230.588,
                262144:  385.104,
                524288:  689.821,
               1048576: 1428.108,
               2097152: 3006.771,
               4194304: 5182.599,
               8388608: 9727.047,
              16777216: 18622.294,
              33554432: 38560.219,
              67108864: 74432.216,
             134217728: 153550.589,
             268435456: 305811.282,
             536870912: 601380.752,
            1073741824: 1198255.900,
        }
        self.p2p_volumes = sorted(self.p2p_volume_to_cost.keys())

        # profiled restart cost
        self.restart_cost_addition = {
            1: 28.760,
            2: 33.587,
            4: 36.189,
            8: 40.882,
            12: 39.020,
            16: 42.892,
            20: 36.749,
            24: 37.849,
            32: 37.588,
        }
        self.restart_cost_removal = {
            1: 54.28,
            2: 56.199,
            4: 60.734,
            8: 69.429,
            12: 75.658,
            16: 81.714,
            20: 83.816,
            24: 92.427,
            32: 95.427,
        }
        self.restart_cost_removal = self.restart_cost_addition
        self.restart_nnodes = sorted(self.restart_cost_addition.keys())

        # proc init time
        if 'gpt' in self.model:
            self.proc_time = 240 # ms
        else:
            self.proc_time = 30 # ms
        # rdzv_time
        self.rdzv_time = 6600 # ms

        self.strategy_speed_throughput = {}
        # some profiling result for gpt-2
        model_info = self.model_signature
        if self.model == 'gpt-1.5b' and model_info['micro_batch_size'] == 1 and model_info['train_batch_size'] == 128:
            self.strategy_speed_throughput = {
                ( 4,  8): (2944.728, 43.468),
                ( 2, 16): (3090.862, 41.412),
                ( 1, 32): (3883.972, 32.956),
                ( 1, 31): (3766.091, 33.987),
                ( 5,  6): (3293.552, 38.864),
                ( 3, 10): (3084.387, 41.499),
                ( 2, 15): (3204.662, 39.942),
                ( 1, 30): (3882.985, 32.964),
                ( 1, 29): (3763.701, 34.009),
                ( 4,  7): (3178.165, 40.275),
                ( 2, 14): (3182.064, 40.225),
                ( 1, 28): (3754.538, 34.092),
                ( 3,  9): (3216.129, 39.799),
                ( 1, 27): (3841.799, 33.318),
                ( 2, 13): (3142.507, 40.732),
                ( 1, 26): (3720.695, 34.402),
                ( 1, 25): (3682.481, 34.759),
                ( 4,  6): (3617.366, 35.385),
                ( 3,  8): (3475.666, 36.827),
                ( 2, 12): (3595.727, 35.598),
                ( 1, 24): (4389.739, 29.159),
                ( 1, 23): (4343.359, 29.470),
                ( 2, 11): (3557.525, 35.980),
                ( 1, 22): (4365.984, 29.318),
                ( 3,  7): (3585.649, 35.698),
                ( 1, 21): (4350.159, 29.424),
                ( 2, 10): (3643.284, 35.133),
                ( 1, 20): (4379.651, 29.226),
                ( 1, 19): (4404.975, 29.058),
                ( 3,  6): (4122.628, 31.048),
                ( 2,  9): (3993.443, 32.053),
                ( 1, 18): (4404.035, 29.064),
                ( 1, 17): (4341.342, 29.484),
                ( 2,  8): (4341.314, 29.484),
                ( 1, 16): (5084.763, 25.173),
                ( 1, 15): (5094.605, 25.125),
                ( 2,  7): (4520.418, 28.316),
                ( 1, 14): (5103.003, 25.083),
                ( 1, 13): (5108.193, 25.058),
                ( 2,  6): (5281.473, 24.236),
                ( 1, 12): (5879.829, 21.769),
                ( 1, 11): (5863.490, 21.830),
                ( 1, 10): (5924.556, 21.605),
                ( 1,  9): (6622.823, 19.327),
                ( 1,  8): (7396.008, 17.307),
                ( 1,  7): (7431.253, 17.225),
                ( 1,  6): (8836.921, 14.485),
            }

            # estimate model build cost
            volume = 50257 * 1024 * 4
            comm_cost = self.interploate(self.p2p_volumes, volume, self.p2p_volume_to_cost) / 1000
            self.model_build_cost = {
                pp: comm_cost + 430 * math.ceil(48 / pp) for pp in range(6, 49)
            }
        elif self.model == 'gpt-6.7b' and model_info['micro_batch_size'] == 1 and model_info['train_batch_size'] == 64:
            self.strategy_speed_throughput = {
                ( 2, 16): (8955.162, 7.147),
                ( 2, 15): (5809.847, 11.016),
                ( 2, 14): (5788.683, 11.056),
                ( 2, 13): (5796.704, 11.041),
                ( 2, 12): (5768.424, 11.095),
                ( 1, 16): (7951.085, 8.049),
                ( 1, 15): (8270.736, 7.738),
                ( 1, 14): (8278.019, 7.731),
                ( 1, 13): (8179.646, 7.824),
                ( 1, 12): (8213.008, 7.793),
            }

            # estimate model build cost in ms
            self.model_build_cost = {
                12: 7_961,
                13: 6_911,
                14: 6_926,
                15: 7_002,
                16: 7_195,
                18: 4_734,
                20: 5_245,
                22: 4_818,
                24: 4_798,
                26: 4_932,
            }

            self.restart_extra_cost = 2_000
        elif self.model == 'bert' and model_info['micro_batch_size'] == 8 and model_info['train_batch_size'] == 1024:
            self.strategy_speed_throughput = {
                (16,  2): (1928.780, 530.905),
                ( 8,  4): (1842.692, 555.709),
                ( 4,  8): (2239.956, 457.152),
                ( 2, 16): (4195.811, 244.053),
                (15,  2): (1976.095, 518.194),
                (10,  3): (1882.224, 544.037),
                ( 6,  5): (2052.494, 498.905),
                ( 5,  6): (2101.000, 487.387),
                ( 3, 10): (3098.485, 330.484),
                (14,  2): (2095.227, 488.730),
                ( 7,  4): (2082.496, 491.718),
                ( 4,  7): (2617.539, 391.207),
                ( 2, 14): (4041.531, 253.369),
                ( 9,  3): (2032.958, 503.700),
                (13,  2): (2097.290, 488.249),
                ( 5,  5): (2390.736, 428.320),
                (12,  2): (2157.026, 474.728),
                ( 8,  3): (2126.731, 481.490),
                ( 6,  4): (2238.342, 457.482),
                ( 4,  6): (2457.851, 416.624),
                ( 3,  8): (2786.690, 367.461),
                ( 2, 12): (3664.781, 279.416),
                (11,  2): (2292.430, 446.688),
                ( 7,  3): (2388.554, 428.711),
                ( 3,  7): (3320.790, 308.360),
                (10,  2): (2406.792, 425.463),
                ( 5,  4): (3177.684, 322.247),
                ( 4,  5): (4821.904, 212.364),
                ( 2, 10): (7588.531, 134.940),
                ( 9,  2): (4890.962, 209.366),
                ( 6,  3): (5239.774, 195.428),
                ( 3,  6): (6213.866, 164.793),
                ( 8,  2): (3697.554, 276.940),
                ( 4,  4): (2924.602, 350.133),
                ( 2,  8): (3739.280, 273.850),
                ( 1, 16): (7517.714, 136.212),
                ( 5,  3): (2936.808, 348.678),
                ( 3,  5): (3375.818, 303.334),
                ( 7,  2): (3070.357, 333.512),
                ( 2,  7): (4420.231, 231.662),
                ( 1, 14): (6665.442, 153.628),
                ( 6,  2): (3380.595, 302.905),
                ( 4,  3): (3427.963, 298.720),
                ( 3,  4): (3662.597, 279.583),
                ( 2,  6): (4188.049, 244.505),
                ( 1, 12): (6382.114, 160.448),
                ( 5,  2): (3975.457, 257.580),
                ( 2,  5): (4706.355, 217.578),
                ( 1, 10): (7837.865, 130.648),
                ( 3,  3): (4400.076, 232.723),
                ( 4,  2): (4421.307, 231.606),
                ( 2,  4): (5109.908, 200.395),
                ( 1,  8): (6809.831, 150.371),
                ( 1,  7): (8352.620, 122.596),
                ( 3,  2): (5686.611, 180.072),
                ( 2,  3): (6145.639, 166.622),
                ( 1,  6): (7904.890, 129.540),
                ( 1,  5): (8655.977, 118.300),
                ( 2,  2): (7892.681, 129.740),
                ( 1,  4): (9703.712, 105.527),
                ( 1,  3): (11482.680, 89.178),
                ( 1,  2): (15020.448, 68.174),
            }
            self.model_build_cost = {
                2: 1752,
                3: 1123,
                4: 1095,
                5: 854,
                6: 860,
                7: 767,
                8: 754,
                10: 633,
                12: 626,
                14: 505,
                16: 531,
            }
        elif self.model == 'resnet152' and model_info['micro_batch_size'] == 32 and model_info['train_batch_size'] == 2048:
            self.strategy_speed_throughput = {
                (16,  2): (653.656, 3133.148),
                ( 8,  4): (810.049, 2528.243),
                ( 4,  8): (1598.497, 1281.204),
                ( 2, 16): (8870.886, 230.868),
                (15,  2): (1133.822, 1806.280),
                ( 5,  6): (1369.234, 1495.727),
                ( 3, 10): (2550.251, 803.058),
                (14,  2): (754.400, 2714.738),
                ( 7,  4): (998.178, 2051.738),
                ( 2, 14): (4889.175, 418.885),
                (13,  2): (1057.509, 1936.626),
                (12,  2): (907.552, 2256.621),
                ( 6,  4): (1074.642, 1905.750),
                ( 4,  6): (1677.455, 1220.897),
                ( 3,  8): (2109.949, 970.639),
                ( 2, 12): (3317.650, 617.305),
                (11,  2): (1022.771, 2002.403),
                (10,  2): (2691.287, 760.974),
                ( 5,  4): (1986.311, 1031.057),
                ( 2, 10): (11302.191, 181.204),
                ( 9,  2): (1078.618, 1898.726),
                ( 3,  6): (6142.403, 333.420),
                ( 8,  2): (1059.615, 1932.777),
                ( 4,  4): (1434.724, 1427.452),
                ( 2,  8): (5517.274, 371.198),
                ( 1, 16): (9759.287, 209.851),
                ( 7,  2): (1302.532, 1572.322),
                ( 1, 14): (32943.566, 62.167),
                ( 6,  2): (1518.270, 1348.904),
                ( 3,  4): (1984.695, 1031.896),
                ( 2,  6): (3298.679, 620.855),
                ( 1, 12): (6878.105, 297.756),
                ( 5,  2): (1922.970, 1065.019),
                ( 1, 10): (35107.824, 58.335),
                ( 4,  2): (2955.415, 692.965),
                ( 2,  4): (3835.129, 534.011),
                ( 1,  8): (7881.612, 259.845),
                ( 3,  2): (4802.160, 426.475),
                ( 1,  6): (15978.146, 128.175),
                ( 2,  2): (7535.594, 271.777),
                ( 1,  4): (8920.462, 229.585),
                ( 1,  2): (14011.118, 146.170),
            }
            self.model_build_cost = {
                2: 110,
                4: 90,
                6: 133,
                8: 135,
                10: 120,
                12: 55,
                14: 120,
                16: 125,
            }
            self.restart_extra_cost = 3360.374 #ms, CIFAR
        elif self.model == 'vgg19' and model_info['micro_batch_size'] == 32 and model_info['train_batch_size'] == 2048:
            self.strategy_speed_throughput = {
                (16,  2): (934.320, 2191.969),
                ( 8,  4): (1676.564, 1221.546),
                ( 4,  8): (3063.508, 668.515),
                (15,  2): (1107.182, 1849.740),
                (10,  3): (1464.817, 1398.127),
                ( 6,  5): (3550.522, 576.817),
                ( 5,  6): (4124.107, 496.592),
                (14,  2): (1140.193, 1796.187),
                ( 7,  4): (2126.479, 963.094),
                ( 4,  7): (5051.454, 405.428),
                ( 9,  3): (1708.575, 1198.660),
                ( 3,  9): (3831.677, 534.492),
                (13,  2): (1223.350, 1674.091),
                ( 5,  5): (4424.361, 462.892),
                (12,  2): (1353.433, 1513.189),
                ( 8,  3): (1680.982, 1218.336),
                ( 6,  4): (2261.728, 905.502),
                ( 4,  6): (5177.933, 395.525),
                ( 3,  8): (3981.333, 514.401),
                (11,  2): (1374.094, 1490.436),
                ( 7,  3): (1960.802, 1044.470),
                ( 3,  7): (6722.477, 304.650),
                (10,  2): (1461.242, 1401.547),
                ( 5,  4): (2557.545, 800.768),
                ( 4,  5): (5038.254, 406.490),
                ( 9,  2): (1578.952, 1297.063),
                ( 6,  3): (2049.112, 999.457),
                ( 3,  6): (6716.539, 304.919),
                ( 2,  9): (5332.671, 384.048),
                ( 8,  2): (1554.607, 1317.375),
                ( 4,  4): (2982.630, 686.642),
                ( 2,  8): (5691.896, 359.810),
                ( 5,  3): (2321.948, 882.018),
                ( 3,  5): (6810.109, 300.729),
                ( 7,  2): (1802.591, 1136.142),
                ( 2,  7): (9527.675, 214.953),
                ( 6,  2): (1896.590, 1079.833),
                ( 4,  3): (2701.110, 758.207),
                ( 3,  4): (3887.169, 526.862),
                ( 2,  6): (9941.914, 205.997),
                ( 5,  2): (2072.841, 988.016),
                ( 2,  5): (9733.998, 210.397),
                ( 3,  3): (3589.008, 570.631),
                ( 4,  2): (2370.673, 863.890),
                ( 2,  4): (5605.076, 365.383),
                ( 3,  2): (3024.927, 677.041),
                ( 2,  3): (5004.662, 409.218),
                ( 2,  2): (4024.426, 508.892),
            }

            self.model_build_cost = {
                2: 110,
                3: 94,
                4: 90,
                5: 105,
                6: 133,
                7: 90,
                8: 120,
                9: 106,
            }
            self.restart_extra_cost = 2_000 #ms, CIFAR + engine
        else:
            # load simulation result of different models
            if self.model_signature is not None:
                strategy_cache = {}
                if os.path.exists(DEFAULT_STRATEGY_CACHE_PATH):
                    with open(DEFAULT_STRATEGY_CACHE_PATH, 'r') as fp:
                        tmp_cache = json.load(fp)
                        for signature_key, oracle in tmp_cache.items():
                            strategy_cache[signature_key] = {}
                            for cfg, value in oracle.items():
                                dp, pp = cfg.split('-')
                                strategy_cache[signature_key][(int(dp), int(pp))] = value

                try:
                    key = json.dumps(self.model_signature)
                    for key, value in strategy_cache[key].items():
                        # key: (dp, pp), value: (speed, throughput, partition)
                        self.strategy_speed_throughput[key] = (value[0], value[1])
                except:
                    print(f'No strategy cache for model: {self.model}. Please run simulation first.')

    def estimate_strategy_performance(self, strategy):
        if strategy in self.strategy_speed_throughput:
            return self.strategy_speed_throughput[(strategy[0] * GPUS_PER_NODE, strategy[1])]
        return (INF, 0)

    def best_strategy(self, world_size):
        if world_size in self.best_strategy_of_nnodes:
            return self.best_strategy_of_nnodes[world_size]

        best_strategy, best_throughput = None, 0
        for dp in range(1, world_size + 1):
            for pp in range(1, world_size + 1):
                if dp * pp <= world_size:
                    strategy = (dp, pp)
                    _, throughput = self.estimate_strategy_performance(strategy)
                    if throughput > best_throughput:
                        best_strategy, best_throughput = strategy, throughput

        self.best_strategy_of_nnodes[world_size] = best_strategy
        return best_strategy

    def all_reduce_cost(self, volume, world_size):
        """
        args:
            volume: in bytes
        return:
            cost: in ms
        """
        return (volume / MB / (10 / 8)) * (2 * (world_size - 1) / world_size)

    def p2p_cost(self, volume):
        """
        args:
            volume: in bytes
        return:
            cost: in ms
        """
        idx = bisect.bisect_left(self.p2p_volumes, volume)
        if idx == 0:
            cost = self.p2p_volume_to_cost[self.p2p_volumes[0]]
        elif idx >= len(self.p2p_volumes):
            cost = (volume / self.p2p_volumes[-1]) * self.p2p_volume_to_cost[self.p2p_volumes[-1]]
        else:
            left = self.p2p_volumes[idx - 1]
            right = self.p2p_volumes[idx]
            cost = self.p2p_volume_to_cost[left] + (self.p2p_volume_to_cost[right] - self.p2p_volume_to_cost[left]) * (volume - left) / (right - left)
        return cost / 1000

    def interploate(self, elements, x, res_dict):
        idx = bisect.bisect_left(elements, x)
        if idx == 0:
            cost = res_dict[elements[0]]
        elif idx >= len(elements):
            cost = (x / elements[-1]) * res_dict[elements[-1]]
        else:
            left = elements[idx - 1]
            right = elements[idx]
            cost = res_dict[left] + (res_dict[right] - res_dict[left]) * (x - left) / (right - left)
        return cost

    def estimate_intra_stage_migration(self, dp, pp):
        group_time = self.estimate_rebuild_comm_group_cost(dp, pp, type='remove')
        total_time = group_time + self.proc_time
        return total_time

    def estimate_rebuild_comm_group_cost(self, dp, pp, type='add'):
        dp = GPUS_PER_NODE * dp
        init_time = 35 # ms
        if self.model in ['gpt-1.5b', 'gpt-6.7b']:
            num_groups = 0
            num_groups = dp + pp + 1 + dp * pp
            if type == 'add':
                group_time = num_groups * 500 # ms
            else:
                if self.model == 'gpt-6.7b':
                    group_time = num_groups * 115.6 # ms
                else:
                    group_time = num_groups * 39.3 # ms
        elif self.model in ['vgg19']:
            if type == 'add':
                group_time = 440 * dp * pp # ms
            else:
                group_time = 340 * dp * pp # ms
        else:
            if dp >= 12:
                group_time = 2000 # ms
            elif dp >= 6:
                group_time = 1500 # ms
            else:
                group_time = 1000 # ms
            if type == 'add':
                group_time = group_time * 1.5
        return init_time + group_time

    def estimate_restart_cost(self, dp, pp, type='no-op'):
        cost = 0
        if self.restart_cost is None:
            if type == 'add':
                cost += self.rdzv_time
                if self.model in ['gpt-6.7b', 'resnet152', 'vgg19']:
                    cost += self.restart_extra_cost
            cost += self.estimate_rebuild_comm_group_cost(dp, pp, type=type)
            cost += self.proc_time
            cost += self.model_build_cost[pp]
        else:
            cost = self.restart_cost * 1000 # ms
        return cost

    def estimate_concurrent_restart_cost(self, dp, pp, type='no-op'):
        cost = self.estimate_restart_cost(dp, pp, type)
        return cost # ms
