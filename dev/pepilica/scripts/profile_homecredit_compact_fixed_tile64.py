"""Homecredit CompactFixedSlotMLP profile with 64x64 GEMM tiles."""
import json
import runpy
import sys
from pathlib import Path

from dev.pepilica.scripts.compact_fixed_tile64 import install

install()
runpy.run_module('dev.pepilica.scripts.profile_homecredit_compact_fixed', run_name='__main__')
report_path = Path(sys.argv[sys.argv.index('--output') + 1]) / 'report.json'
report = json.loads(report_path.read_text())
report['tiles'] = '64x64'
report_path.write_text(json.dumps(report, indent=2) + '\n')
