"""Controlled CUDA graph comparison for full-pack fixed-slot buffer fills."""
import json
import random
import statistics
from pathlib import Path

import torch
from dev.pepilica.scripts.compact_fixed_tile64 import install as install_tiles
from dev.pepilica.scripts.experiment_fixed_nozero import install as nozero, uninstall as original
from dev.pepilica.scripts.benchmark_active_fixed_graph import run


def main():
    torch.cuda.set_per_process_memory_fraction(.3)
    torch.set_num_threads(2)
    torch.autograd.graph.set_override_stale_capture_stream(True)
    install_tiles()
    cfg=json.loads(Path('dev/pepilica/configs/churn.json').read_text())
    rng=random.Random(cfg['seed'])
    widths=tuple(rng.choice(cfg['widths']) for _ in range(cfg['pack_size']))
    depths=tuple(rng.randint(1,cfg['depth']) for _ in widths)
    active=tuple(range(len(widths)))
    orders=[('original','nozero','active'),('active','original','nozero'),('nozero','active','original')]
    samples=[]
    for iteration, order in enumerate(orders):
        for kind in order:
            original()
            if kind=='nozero':
                nozero()
            torch.manual_seed(2026)
            r=run(widths,depths,cfg['batch_size'],active, 'active' if kind=='active' else 'fixed')
            r.update(variant=kind,repeat=iteration)
            samples.append(r)
            torch.cuda.empty_cache()
    original()
    summary={k:{name:statistics.median(x[name] for x in samples if x['variant']==k)
                for name in ('forward_ms','backward_ms')}
             for k in ('original','nozero','active')}
    out=Path('.cache/benchmarks/active-slots-churn-20261106/nozero-control.json')
    out.write_text(json.dumps({'gpu':torch.cuda.get_device_name(),
                               'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),
                               'scope':'full active pack; isolated CUDA graph replays',
                               'samples':samples,'median':summary},indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
