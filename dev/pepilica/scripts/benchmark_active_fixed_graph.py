"""CUDA-graph timings of churn-shaped active and fixed slot forward/backward."""
import json
import random
import statistics
from pathlib import Path

import torch
from dev.pepilica.scripts.compact_fixed_tile64 import install
from tabpack_experimental.active_fixed_slot import ActiveFixedSlotMLP
from tabpack_experimental.compact_mlp import CompactFixedSlotMLP


def run(widths,depths,batch,active,kind):
    cls=ActiveFixedSlotMLP if kind=='active' else CompactFixedSlotMLP
    model=cls(widths,batch,depths,device='cuda',dtype=torch.float32,dropout=0.1)
    x=torch.randn(batch,sum(widths[i] for i in active) if kind=='active' else sum(widths),
                  device='cuda',dtype=torch.bfloat16,requires_grad=True)
    for _ in range(4):
        model.zero_grad(set_to_none=True)
        x.grad=None
        y=model(x,active)
        y.backward(torch.ones_like(y))
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    x.grad=None
    forward=torch.cuda.CUDAGraph()
    with torch.cuda.graph(forward):
        y=model(x,active)
    dy=torch.ones_like(y)
    backward=torch.cuda.CUDAGraph()
    with torch.cuda.graph(backward):
        y.backward(dy,retain_graph=True)
    for _ in range(3):
        forward.replay()
        backward.replay()
    torch.cuda.synchronize()
    samples=[]
    for _ in range(15):
        events=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
        events[0].record()
        for _ in range(20): forward.replay()
        events[1].record()
        for _ in range(20): backward.replay()
        events[2].record()
        torch.cuda.synchronize()
        samples.append((events[0].elapsed_time(events[1])/20,events[1].elapsed_time(events[2])/20))
    result={'active_count':len(active),'kind':kind,'forward_ms':statistics.median(s[0] for s in samples),
            'backward_ms':statistics.median(s[1] for s in samples)}
    print(json.dumps(result),flush=True)
    return result


def main():
    torch.cuda.set_per_process_memory_fraction(.3)
    torch.set_num_threads(2)
    torch.autograd.graph.set_override_stale_capture_stream(True)
    install()
    cfg=json.loads(Path('dev/pepilica/configs/churn.json').read_text())
    rng=random.Random(cfg['seed'])
    widths=tuple(rng.choice(cfg['widths']) for _ in range(cfg['pack_size']))
    depths=tuple(rng.randint(1,cfg['depth']) for _ in widths)
    ids=list(range(len(widths)))
    rng.shuffle(ids)
    result=[]
    for n in (64,32,8):
        active=tuple(sorted(ids[:n]))
        for kind in ('fixed','active'):
            result.append(run(widths,depths,cfg['batch_size'],active,kind))
    Path('.cache/benchmarks/active-slots-churn-20261106/graph.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
