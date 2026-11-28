"""Run review regressions and record active-set workspace peak memory on CUDA."""

import json
import runpy
import sys
from pathlib import Path

import pytest
import torch

from tabpackage._muon_grouped import GroupedNewtonSchulz, extension


def main():
    sys.path.insert(0, str(Path.cwd()))
    out = Path('../output')
    out.mkdir(exist_ok=True)
    code = pytest.main(
        [
            'tests/test_muon_review.py',
            'common/tests',
            '-q',
            f'--junitxml={out / "pytest.xml"}',
        ]
    )
    if code:
        raise SystemExit(code)
    runpy.run_path(
        'dev/veleboks/scripts/check_rectangular_muon.py', run_name='__main__'
    )
    torch.manual_seed(0)
    x = torch.zeros(16, 256, 8192, device='cuda')
    widths = torch.full((16,), 256, device='cuda', dtype=torch.long)
    sets = [
        torch.arange(0, 16, 2, device='cuda'),
        torch.arange(1, 16, 2, device='cuda'),
    ]
    runtime = GroupedNewtonSchulz()

    def call(active):
        x.zero_()
        x[active] = torch.randn(len(active), 256, 8192, device='cuda')
        result = runtime(x, widths, active_pack_idx=active, rectangular=True)
        inactive = sets[1] if active is sets[0] else sets[0]
        assert result[inactive].count_nonzero() == 0
        torch.cuda.synchronize()

    call(sets[0])
    plan = runtime.plan
    torch.cuda.reset_peak_memory_stats()
    call(sets[0])
    steady_peak = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for i in range(8):
        call(sets[(i + 1) % 2])
        assert runtime.plan is plan
    reuse_peak = torch.cuda.max_memory_allocated()
    # Isolate the previous problem: a second plan's construction while the old
    # buffers remain live. Measure this separately from normalization temporaries.
    live_before_rebuild = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    replacement = extension().Plan(x, widths.tolist(), list(range(16)), True)
    torch.cuda.synchronize()
    rebuild_peak = torch.cuda.max_memory_allocated()
    del replacement
    report = {
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(),
        'shape': list(x.shape),
        'alternating_active_sets': 8,
        'steady_peak_bytes': steady_peak,
        'reuse_peak_bytes': reuse_peak,
        'overlapping_plan_construction_peak_bytes': rebuild_peak,
        'extra_live_plan_bytes': rebuild_peak - live_before_rebuild,
        'inactive_updates_zero': True,
    }
    (out / 'workspace-memory.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    assert reuse_peak <= steady_peak + 1024**2
    assert rebuild_peak - live_before_rebuild >= 128 * 1024**2


if __name__ == '__main__':
    main()
