import copy
from collections.abc import Callable

import torch
from torch import Tensor, nn

from tabpackage.nn import TabPack


class WidthBucketedModel(nn.Module):
    inverse_order: Tensor

    def __init__(self, source: TabPack, model_config: dict):
        super().__init__()
        if model_config.get('numerical_embeddings') is not None:
            raise ValueError('Width bucketing currently supports the base model only')
        pack_size = source.pack_size
        widths = model_config['d_block']
        if isinstance(widths, int):
            widths = [widths] * pack_size
        self._pack_size = pack_size
        self.widths = sorted(set(widths))
        self.buckets = nn.ModuleList()
        self.reference_muon_shapes = [
            tuple(block.linear.weight.shape[-2:]) for block in source.backbone.blocks
        ]
        device = next(source.parameters()).device
        original_order = []
        source_parameters = dict(source.named_parameters())
        for bucket_id, width in enumerate(self.widths):
            members = [i for i, w in enumerate(widths) if w == width]
            indices = torch.tensor(members, device=device)
            self.register_buffer(f'members_{bucket_id}', indices)
            original_order.extend(members)
            config = copy.deepcopy(model_config)
            config['pack_size'] = len(members)
            config['d_block'] = width
            config.pop('max_d_block', None)
            for key in ('n_blocks', 'dropout'):
                if isinstance(config[key], list):
                    config[key] = [config[key][i] for i in members]
            bucket = TabPack(**config).to(device).train(source.training)
            with torch.no_grad():
                for name, parameter in bucket.named_parameters():
                    selected = source_parameters[name].index_select(0, indices)
                    slices = (
                        slice(None),
                        *(slice(size) for size in parameter.shape[1:]),
                    )
                    parameter.copy_(selected[slices])
            self.buckets.append(bucket)
        inverse = torch.tensor(original_order, device=device).argsort()
        self.register_buffer('inverse_order', inverse)
        self.train(source.training)

    @property
    def pack_size(self):
        return self._pack_size

    def member_indices(self, bucket_id: int) -> Tensor:
        return getattr(self, f'members_{bucket_id}')

    def forward(self, x_num, x_cat, pack_idx=None):
        if pack_idx is not None:
            raise ValueError('External member indexing is not supported')
        outputs = []
        for bucket_id, bucket in enumerate(self.buckets):
            indices = self.member_indices(bucket_id)
            outputs.append(
                bucket(
                    x_num.index_select(0, indices)
                    if x_num is not None and x_num.ndim == 3
                    else x_num,
                    x_cat.index_select(0, indices)
                    if x_cat is not None and x_cat.ndim == 3
                    else x_cat,
                )
            )
        return torch.cat(outputs, dim=0).index_select(0, self.inverse_order)


class BucketedOptimizer:
    def __init__(
        self,
        model: WidthBucketedModel,
        optimizer_config: dict,
        make_optimizer: Callable,
    ):
        self.optimizers = []
        for bucket_id, bucket in enumerate(model.buckets):
            members = model.member_indices(bucket_id).tolist()
            config = {
                key: [value[i] for i in members] if isinstance(value, list) else value
                for key, value in optimizer_config.items()
            }
            self.optimizers.append(
                make_optimizer(
                    bucket,
                    config,
                    muon_reference_shapes=model.reference_muon_shapes,
                )
            )

    @property
    def param_groups(self):
        return [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self):
        return {'buckets': [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state):
        if len(state['buckets']) != len(self.optimizers):
            raise ValueError('Optimizer checkpoint has a different bucket count')
        for optimizer, bucket_state in zip(self.optimizers, state['buckets']):
            optimizer.load_state_dict(bucket_state)
