"""Compile and probe the installed cuBLAS with unchanged Muon BF16 precision."""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from tabpackage._muon_grouped import GroupedNewtonSchulz
from tabpackage.optim import _zeropower_via_newtonschulz5

os.environ['PATH'] = str(Path.home() / 'tabpack/.venv/bin') + ':' + os.environ['PATH']
print(
    'Visible devices:',
    [
        (
            i,
            torch.cuda.get_device_name(i),
            str(torch.cuda.get_device_properties(i).uuid),
        )
        for i in range(torch.cuda.device_count())
    ],
    flush=True,
)
parser = argparse.ArgumentParser()
parser.add_argument('--device-index', type=int, default=0)
args = parser.parse_args()
torch.cuda.set_device(args.device_index)
out = Path('../output')
out.mkdir(exist_ok=True)
report = {
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(),
    'capability': torch.cuda.get_device_capability(),
}
t0 = time.perf_counter()
try:
    widths = torch.tensor([32, 64, 128, 256, 384, 512], device='cuda')
    g = torch.zeros(6, 512, 512, device='cuda')
    torch.manual_seed(42)
    for i, w in enumerate(widths.tolist()):
        g[i, :w, :w] = torch.randn(w, w, device='cuda')
    backend = GroupedNewtonSchulz()
    result = backend(g, widths).clone()
    torch.cuda.synchronize()
    report['setup_first_call_seconds'] = time.perf_counter() - t0
    report['cublas_version'] = backend.plan.version
    baseline = _zeropower_via_newtonschulz5(g, 5)
    delta = result.float() - baseline.float()
    report.update(
        max_abs=delta.abs().max().item(),
        relative_l2=(delta.norm() / baseline.float().norm()).item(),
        finite=bool(result.isfinite().all()),
    )
except Exception as exc:  # noqa: BLE001 -- record unsupported hardware/build errors
    report['error'] = str(exc)
    report['setup_failed_seconds'] = time.perf_counter() - t0
(out / 'probe.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
