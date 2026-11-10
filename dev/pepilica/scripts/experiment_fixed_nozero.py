"""Scoped benchmark variant: omit redundant activation fills inside CompactFixedSlotMLP.

The forward intermediate has undefined inactive entries, so this must only be
used when consumed by the compact block's masked pointwise path.
"""
import torch
from torch.autograd.function import once_differentiable
import tabpack_experimental.compact_mlp as compact

original_forward = compact._FixedSlotLinear.forward
original_backward = compact._FixedSlotCompactLayer.backward


def forward(ctx, x, weight, bias, plan):
    output = torch.empty_like(x) if plan.active else torch.zeros_like(x)
    if plan.active:
        compact.fixed_slot_mm[(len(plan.forward_tasks),)](
            x, weight, bias, output, plan.descriptors, plan.forward_tasks,
            plan.stride, plan.batch, 32, 32, 32, num_warps=4)
    ctx.save_for_backward(x, weight)
    ctx.plan = plan
    return output


@once_differentiable
def backward(ctx, grad_output):
    x, weight, output, seed = ctx.saved_tensors
    block, plan = ctx.block, ctx.plan
    grad_output = grad_output.contiguous()
    dz = torch.empty_like(grad_output)
    block.pointwise(grad_output, output, dz, 'activation_grad', seed, ctx.dropout)
    dx = torch.empty_like(x) if ctx.needs_input_grad[0] else None
    dw = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
    db = torch.zeros(plan.stride, device=x.device, dtype=x.dtype) if ctx.needs_input_grad[2] else None
    if dx is not None:
        compact.fixed_slot_dx[(len(plan.forward_tasks),)](
            dz, weight, dx, plan.descriptors, plan.forward_tasks,
            plan.stride, plan.batch, 32, 32, 32, num_warps=4)
        if block.has_inactive:
            block.pointwise(grad_output, output, dx, 'carry_grad')
    if dw is not None:
        compact.fixed_slot_dw[(len(plan.weight_tasks),)](
            dz, x, dw, plan.descriptors, plan.weight_tasks,
            plan.stride, plan.batch, 32, 32, 32, num_warps=4)
    if db is not None:
        compact.fixed_slot_db[(len(plan.bias_tasks),)](
            dz, db, plan.descriptors, plan.bias_tasks,
            plan.stride, plan.batch, 128, 32, num_warps=4)
    return dx, dw, db, None, None, None


def install():
    compact._FixedSlotLinear.forward = staticmethod(forward)
    compact._FixedSlotCompactLayer.backward = staticmethod(backward)


def uninstall():
    compact._FixedSlotLinear.forward = staticmethod(original_forward)
    compact._FixedSlotCompactLayer.backward = staticmethod(original_backward)
