"""Benchmark churn-shaped hidden blocks with fixed versus active-only activations."""
import argparse
import gc
import json
import random
import statistics
import time
from pathlib import Path

import torch
from dev.pepilica.scripts.compact_fixed_tile64 import install
from tabpack_experimental.active_fixed_slot import ActiveFixedSlotMLP
from tabpack_experimental.compact_mlp import CompactFixedSlotMLP


def sync():
    torch.cuda.synchronize()


def benchmark(widths, depths, batch, active, mode, steps):
    torch.manual_seed(42)
    cls = ActiveFixedSlotMLP if mode == 'active' else CompactFixedSlotMLP
    model = cls(widths, batch, depths, device='cuda', dtype=torch.float32, dropout=0.1)
    xwidth = sum(widths[i] for i in active) if mode == 'active' else sum(widths)
    x = torch.randn(batch, xwidth, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    if mode == 'active':
        model.prepare(active)
    def run(timed):
        model.zero_grad(set_to_none=True)
        x.grad = None
        if timed:
            events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            events[0].record()
        y = model(x, active)
        if timed:
            events[1].record()
        y.backward(torch.ones_like(y))
        if timed:
            events[2].record()
            return events
    for _ in range(4):
        run(False)
    sync()
    model.zero_grad(set_to_none=True)
    x.grad = None
    sync()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples=[]
    for _ in range(steps):
        marks=run(True)
        sync()
        samples.append((marks[0].elapsed_time(marks[1]), marks[1].elapsed_time(marks[2]), marks[0].elapsed_time(marks[2])))
    peak=torch.cuda.max_memory_allocated()
    result=dict(mode=mode, active_count=len(active), active_width=sum(widths[i] for i in active),
                total_width=sum(widths), batch=batch,
                forward_ms=statistics.median(v[0] for v in samples),
                backward_ms=statistics.median(v[1] for v in samples),
                forward_backward_ms=statistics.median(v[2] for v in samples),
                peak_allocated_bytes=peak, persistent_allocated_bytes=baseline,
                peak_increment_bytes=peak-baseline)
    del model,x
    gc.collect()
    torch.cuda.empty_cache()
    return result


def transition(widths, depths, batch, active, steps):
    model=ActiveFixedSlotMLP(widths,batch,depths,device='cuda',dtype=torch.float32,dropout=0.1)
    full=tuple(range(len(widths)))
    x=torch.randn(batch,sum(widths),device='cuda',dtype=torch.bfloat16)
    sync()
    start=time.perf_counter()
    model.prepare(active)
    sync()
    plan_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter()
    packed=model.repack(x,full,active)
    sync()
    first_pack_ms=(time.perf_counter()-start)*1000
    packed=model.repack(x,full,active)
    sync()
    times=[]
    for _ in range(steps):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        packed=model.repack(x,full,active)
        b.record()
        sync()
        times.append(a.elapsed_time(b))
    result={'active_count':len(active),'metadata_cold_wall_ms':plan_ms,
            'repack_cold_wall_ms':first_pack_ms,'repack_cached_gpu_ms':statistics.median(times),
            'repacked_bytes':packed.numel()*packed.element_size()}
    del model,x,packed
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=30)
    args=p.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.3)
    torch.set_num_threads(2)
    install()
    cfg=json.loads(Path('dev/pepilica/configs/churn.json').read_text())
    rng=random.Random(cfg['seed'])
    widths=tuple(rng.choice(cfg['widths']) for _ in range(cfg['pack_size']))
    depths=tuple(rng.randint(1,cfg['depth']) for _ in widths)
    permutation=list(range(len(widths)))
    rng.shuffle(permutation)
    scenarios=[tuple(sorted(permutation[:n])) for n in (64,32,8)]
    results=[]
    for ids in scenarios:
        for mode in ('fixed','active'):
            r=benchmark(widths,depths,cfg['batch_size'],ids,mode,args.steps)
            results.append(r)
            print(json.dumps(r),flush=True)
    transitions=[]
    for ids in scenarios[1:]:
        r=transition(widths,depths,cfg['batch_size'],ids,args.steps)
        transitions.append(r)
        print(json.dumps(r),flush=True)
    report={'scope':'Synthetic churn-config hidden stack only, no real examples, no optimizer',
            'gpu':torch.cuda.get_device_name(), 'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),
            'dtype':'BF16 activations, FP32 parameters', 'tile':64, 'steps':args.steps,
            'widths':widths,'depths':depths,'active_ids':scenarios,'results':results,'transitions':transitions}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
