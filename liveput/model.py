import numpy as np

from cost_cfg import get_model_layer_costs


MB = 1024 * 1024


def gen_model_signature(model, nnodes, ngpu_per_node):
    signature = {
        'nnodes': nnodes,
        'ngpu_per_node': ngpu_per_node,
        'name': model.name,
        'train_batch_size': model.train_batch_size,
        **model.arch_cfg,
    }
    return signature


class ModelSpec:
    def __init__(self, name, layer_specs, train_batch_size, nparts=0, optimizer='adam', **arch_cfg):
        self.name = name
        self.layer_specs = layer_specs
        self.train_batch_size = train_batch_size
        self.arch_cfg = arch_cfg

        self.optimizer = optimizer.lower()
        self.optim_state_cnt = 2 if self.optimizer == 'adam' else 1

        self.partition(nparts, update=True)

    def partition(self, nparts, update=False):
        # split layers into stages
        if nparts == 0:
            chunk_prefix = [0, len(self.layer_specs)]
            param_cnt = 0
            for spec in self.layer_specs:
                param_cnt += spec['params']
            part_params = [param_cnt]
        else:
            chunk_size = (len(self.layer_specs) - 1) // nparts
            chunks = [chunk_size + 1] + [chunk_size] * (nparts - 1)
            for i in range((len(self.layer_specs) - 1) % nparts):
                chunks[-i-2] += 1

            part_params = []
            chunk_prefix = list(np.cumsum([0] + chunks))
            for i in range(nparts):
                local_start = chunk_prefix[i]
                local_end = chunk_prefix[i+1]

                param_cnt = 0
                for spec in self.layer_specs[local_start:local_end]:
                    param_cnt += spec['params']
                part_params.append(param_cnt)

        if update:
            self.nparts = nparts
            self.parts, self.part_params = chunk_prefix, part_params

        return chunk_prefix, part_params

    def get_layers_params(self, layers):
        return sum([self.layer_specs[i]['params'] for i in layers])

    def __getitem__(self, idx):
        return self.layer_specs[idx]

    def max_pipeline_stages(self):
        if 'gpt' in self.name or 'bert' in self.name:
            return len(self.layer_specs) - 1
        else:
            return len(self.layer_specs)

    def has_embedding(self):
        return 'gpt' in self.name or 'bert' in self.name

    def stage_p2p_volume(self):
        return self.arch_cfg['stage_p2p_num'] * 4 / MB

    def stage_params(self, stage_id):
        return self.part_params[stage_id]

    def stage_optim_states(self, stage_id):
        return self.part_params[stage_id] * self.optim_state_cnt

    def stage_model_states(self, stage_id):
        return self.part_params[stage_id] * (1 + self.optim_state_cnt)

    @staticmethod
    def build(model, train_batch_size, nparts=0, **kwargs):
        if model == 'gpt-2':
            micro_batch_size = kwargs.get('micro_batch_size', 2)
            nlayers = kwargs.get('nlayers', 24)
            seq_length = kwargs.get('seq_length', 1024)
            hidden_size = 1024
            nheads = 16
            vocab_size = 50257
            optimizer = 'adam'

            layer_costs = get_model_layer_costs(model, micro_batch_size=micro_batch_size,
                                                nlayers=nlayers, seq_length=seq_length)
        elif model == 'gpt-1.5b':
            micro_batch_size = kwargs.get('micro_batch_size', 1)
            nlayers = kwargs.get('nlayers', 48)
            seq_length = kwargs.get('seq_length', 1024)
            hidden_size = 1600
            nheads = 25
            vocab_size = 50257
            optimizer = 'adam'

            layer_costs = get_model_layer_costs(model, micro_batch_size=micro_batch_size,
                                                nlayers=nlayers, seq_length=seq_length)
        elif model == 'gpt-2.7b':
            micro_batch_size = kwargs.get('micro_batch_size', 1)
            nlayers = kwargs.get('nlayers', 32)
            seq_length = kwargs.get('seq_length', 1024)
            hidden_size = 2560
            nheads = 32
            vocab_size = 50257
            optimizer = 'adam'

            layer_costs = get_model_layer_costs(model, micro_batch_size=micro_batch_size,
                                                nlayers=nlayers, seq_length=seq_length)
        elif model == 'gpt-6.7b':
            micro_batch_size = kwargs.get('micro_batch_size', 1)
            nlayers = kwargs.get('nlayers', 36)
            seq_length = kwargs.get('seq_length', 1024)
            hidden_size = 4096
            nheads = 32
            vocab_size = 50257
            optimizer = 'adam'

            layer_costs = get_model_layer_costs(model, micro_batch_size=micro_batch_size,
                                                nlayers=nlayers, seq_length=seq_length)
        elif model == 'bert':
            micro_batch_size = kwargs.get('micro_batch_size', 8)
            nlayers = kwargs.get('nlayers', 24)
            seq_length = kwargs.get('seq_length', 512)
            hidden_size = 1024
            nheads = 16
            vocab_size = 30522
            optimizer = 'adam'

            layer_costs = get_model_layer_costs(model, micro_batch_size=micro_batch_size,
                                                nlayers=nlayers, seq_length=seq_length)
        elif model == 'resnet152':
            micro_batch_size = kwargs.get('micro_batch_size', 32)
            optimizer = 'adam'

            layer_costs = get_model_layer_costs(model, micro_batch_size=micro_batch_size)
        elif model == 'vgg19':
            micro_batch_size = kwargs.get('micro_batch_size', 32)
            optimizer = 'adam'

            layer_costs = get_model_layer_costs(model, micro_batch_size=micro_batch_size)
        elif model == 'resnet50':
            micro_batch_size = kwargs.get('micro_batch_size', 64)
            optimizer = 'sgd'

            layer_costs = get_model_layer_costs(model, micro_batch_size=micro_batch_size)
        else:
            raise ValueError('Unknown model: {}'.format(model))

        if 'gpt' in model or 'bert' in model:
            arch_cfg = {
                'micro_batch_size': micro_batch_size,
                'nlayers': nlayers,
                'seq_length': seq_length,
                'hidden_size': hidden_size,
                'nheads': nheads,
                'vocab_size': vocab_size,
                'stage_p2p_num': micro_batch_size * seq_length * hidden_size,
            }
        else:
            arch_cfg = {
                'micro_batch_size': micro_batch_size,
                'stage_p2p_num': 0,
            }

        return ModelSpec(model, layer_costs, train_batch_size, nparts=nparts, optimizer=optimizer, **arch_cfg)
