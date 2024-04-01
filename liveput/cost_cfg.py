import numpy as np


MB = 1024 * 1024

# helper method
def parse_gpt_model():
    params = []
    fw_times, bw_times, opt_times = [], [], []
    acts = []
    with open('log.txt', 'r') as f:
        idx = 0
        for line in f.readlines():
            if 'numel' in line:
                params.append(int(line.split()[-1]))
            elif 'time' in line:
                fw = float(line.split(':')[1].split()[0])
                bw = float(line.split(':')[2].split()[0])
                act = max(0, int(line.split(':')[3].split()[0]))
                fw_times.append(fw)
                bw_times.append(bw)
                acts.append(act)
            elif 'step' in line:
                step = float(line.split()[1]) / 10
    total = sum(params[:-1])
    print(f'forward: {sum(fw_times[:2]):.3f}, backward: {sum(bw_times[:2]):.3f}, optimizer: {step * params[0] / total:.3f}, acts: {sum(acts[:2])}, params: {params[0]}')
    print(f'forward: {np.mean(fw_times[2:-3]):.3f}, backward: {np.mean(bw_times[2:-3]):.3f}, optimizer: {step * np.mean(params[1:-2]) / total:.3f}, acts: {np.mean(acts[2:-2]):.3f}, params: {np.mean(params[1:-2])}')
    print(f'forward: {sum(fw_times[-3:]):.3f}, backward: {sum(bw_times[-3:]):.3f}, optimizer: {step * sum(params[-2:]) / total:.3f}, acts: {sum(acts[-3:])}, params: {sum(params[-2:])}')


def get_model_layer_costs(model, micro_batch_size, nlayers=1, seq_length=1024):
    if model == 'gpt-2':
        # model gpt-2, hidden_size: 1024, nheads: 16
        if micro_batch_size == 2 and seq_length == 1024:
            _profile = {
                0: {
                    'forward': 0.1659,
                    'backward': 1.3790,
                    'optimizer': 5.5431,
                    'acts': 10485760 / MB, # MB
                    'params': 42498048,
                },
                1: {
                    'forward': 8.3165,
                    'backward': 13.9290,
                    'optimizer': 0.8212,
                    'acts': 775979008 / MB, # MB
                    'params': 12596224,
                },
                2: {
                    'forward': 13.2382,
                    'backward': 25.2204,
                    'optimizer': 5.5493,
                    'acts': 663192576 / MB, # MB
                    'params': 42498048,
                },
            }
        elif micro_batch_size == 4 and seq_length == 1024:
            _profile = {
                0: {
                    'forward': 0.2577,
                    'backward': 1.6028,
                    'optimizer': 5.5431,
                    'acts': 20971520 / MB, # MB
                    'params': 42498048,
                },
                1: {
                    'forward': 16.3985,
                    'backward': 27.3141,
                    'optimizer': 0.8212,
                    'acts': 1551958016 / MB, # MB
                    'params': 12596224,
                },
                2: {
                    'forward': 26.4790,
                    'backward': 50.8933,
                    'optimizer': 5.5493,
                    'acts': 1326384128 / MB, # MB
                    'params': 42498048,
                },
            }
        else:
            raise NotImplementedError
    elif model == 'gpt-1.5b':
        if micro_batch_size == 1 and seq_length == 1024:
            _profile = {
                0: {
                    'forward': 0.752,
                    'backward': 0.798,
                    'optimizer': 5.655,
                    'acts': 9420800 / MB, # MB
                    'params': 82124800,
                },
                1: {
                    'forward': 6.510,
                    'backward': 13.448,
                    'optimizer': 2.117,
                    'acts': 282034176 / MB, # MB
                    'params': 30740800,
                },
                2: {
                    'forward': 11.238,
                    'backward': 23.762,
                    'optimizer': 5.656,
                    'acts': 186392576 / MB, # MB
                    'params': 82128000,
                },
            }
        elif micro_batch_size == 2 and seq_length == 1024:
            _profile = {
                0: {
                    'forward': 0.620,
                    'backward': 1.003,
                    'optimizer': 5.655,
                    'acts': 16793600 / MB, # MB
                    'params': 82124800,
                },
                1: {
                    'forward': 12.494,
                    'backward': 26.396,
                    'optimizer': 2.117,
                    'acts': 557575606.857 / MB, # MB
                    'params': 30740800,
                },
                2: {
                    'forward': 22.440,
                    'backward': 47.589,
                    'optimizer': 5.656,
                    'acts': 373817344 / MB, # MB
                    'params': 82128000,
                },
            }
        else:
            raise NotImplementedError
    elif model == 'gpt-2.7b':
        if micro_batch_size == 1 and seq_length == 1024:
            _profile = {
                0: {
                    'forward': 0.625,
                    'backward': 1.015,
                    'optimizer': 7.708,
                    'acts': 13631488 / MB, # MB
                    'params': 131399680,
                },
                1: {
                    'forward': 13.487,
                    'backward': 28.031,
                    'optimizer': 4.615,
                    'acts': 304447488 / MB, # MB
                    'params': 78676480,
                },
                2: {
                    'forward': 17.830,
                    'backward': 37.045,
                    'optimizer': 7.708,
                    'acts': 174596096 / MB, # MB
                    'params': 131404800,
                },
            }
        elif micro_batch_size == 1 and seq_length == 2048:
            _profile = {
                0: {
                    'forward': 0.494,
                    'backward': 1.046,
                    'optimizer': 7.708,
                    'acts': 26214400 / MB, # MB
                    'params': 131399680,
                },
                1: {
                    'forward': 31.163,
                    'backward': 67.552,
                    'optimizer': 4.615,
                    'acts': 1011111253.333 / MB, # MB
                    'params': 78676480,
                },
                2: {
                    'forward': 36.283,
                    'backward': 72.878,
                    'optimizer': 7.708,
                    'acts': 349192192 / MB, # MB
                    'params': 131404800,
                },
            }
        else:
            raise NotImplementedError
    elif model == 'gpt-6.7b':
        if micro_batch_size == 1 and seq_length == 1024:
            _profile = {
                0: {
                    'forward': 0.607,
                    'backward': 1.357,
                    'optimizer': 11.852,
                    'acts': 20971520 / MB, # MB
                    'params': 214433792,
                },
                1: {
                    'forward': 31.861,
                    'backward': 63.118,
                    'optimizer': 11.131,
                    'acts': 281550848 / MB, # MB
                    'params': 201379840,
                },
                2: {
                    'forward': 28.082,
                    'backward': 59.148,
                    'optimizer': 11.852,
                    'acts': 155721728 / MB, # MB
                    'params': 214441984,
                },
            }
        elif micro_batch_size == 1 and seq_length == 2048:
            _profile = {
                0: {
                    'forward': 0.704,
                    'backward': 1.506,
                    'optimizer': 11.852,
                    'acts': 41943040 / MB, # MB
                    'params': 214433792,
                },
                1: {
                    'forward': 68.155,
                    'backward': 140.723,
                    'optimizer': 11.131,
                    'acts': 866140160 / MB, # MB
                    'params': 201379840,
                },
                2: {
                    'forward': 58.414,
                    'backward': 117.811,
                    'optimizer': 11.852,
                    'acts': 311443456 / MB, # MB
                    'params': 214441984,
                },
            }
        else:
            raise NotImplementedError
    elif model == 'bert':
        if micro_batch_size == 8:
            _profile = {
                0: {
                    'forward': 140.248, # not used, randomly set
                    'backward': 272.695, # not used, randomly set
                    'optimizer': 8.359, # not used, randomly set
                    'acts': 420.995, # MB
                    'params': 31782912,
                },
                1: {
                    'forward': 140.248, # not used, randomly set
                    'backward': 272.695, # not used, randomly set
                    'optimizer': 8.359, # not used, randomly set
                    'acts': 278.96, # MB
                    'params': 12596224,
                },
                2: {
                    'forward': 140.248, # not used, randomly set
                    'backward': 272.695, # not used, randomly set
                    'optimizer': 8.359, # not used, randomly set
                    'acts': 420.995, # MB
                    'params': 31782912,
                }
            }
    elif model == 'resnet152':
        if micro_batch_size == 32:
            _profile = {
                0: {
                    'forward': 140.248,
                    'backward': 272.695,
                    'optimizer': 8.359,
                    'acts': 10940.249, # MB
                    'params': 1162876.16, # 60192808,
                },
                1: {
                    'forward': 140.248,
                    'backward': 272.695,
                    'optimizer': 8.359,
                    'acts': 10940.249, # MB
                    'params': 204900, # 60192808,
                }
            }

        else:
            raise NotImplementedError
    elif model == 'vgg19':
        if micro_batch_size == 32:
            _profile = {
                0: {
                    'forward': 140.248,
                    'backward': 272.695,
                    'optimizer': 8.359,
                    'acts': 10940.249, # MB
                    'params': 1251524,
                },
                1: {
                    'forward': 140.248,
                    'backward': 272.695,
                    'optimizer': 8.359,
                    'acts': 10940.249, # MB
                    'params': 102764544,
                },
                2: {
                    'forward': 140.248,
                    'backward': 272.695,
                    'optimizer': 8.359,
                    'acts': 10940.249, # MB
                    'params': 16781312,
                },
                3: {
                    'forward': 140.248,
                    'backward': 272.695,
                    'optimizer': 8.359,
                    'acts': 10940.249, # MB
                    'params': 409700,
                },
            }
        else:
            raise NotImplementedError
    elif model == 'resnet50':
        if micro_batch_size == 64:
            _profile = {
                0: {
                    'forward': 61.121,
                    'backward': 117.517,
                    'optimizer': 1.768,
                    'acts': 5313.936, # MB
                    'params': 25557032,
                }
            }
        else:
            raise NotImplementedError

    model_costs = []
    if 'gpt' in model or 'bert' in model:
        model_costs.append(_profile[0])
        for _ in range(nlayers):
            model_costs.append(_profile[1])
        model_costs.append(_profile[2])
    elif 'vgg19' in model:
        for _ in range(16):
            model_costs.append(_profile[0])
        model_costs.append(_profile[1])
        model_costs.append(_profile[2])
        model_costs.append(_profile[3])
    elif 'resnet152' in model:
        for _ in range(51):
            model_costs.append(_profile[0])
        model_costs.append(_profile[1])
    else:
        model_costs.append(_profile[0])
    return model_costs


# if __name__ == '__main__':
#     parse_gpt_model()
