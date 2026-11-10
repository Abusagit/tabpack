"""Run epoch-3 Maps benchmark with 64x64 fixed-slot and Muon GEMMs."""

import runpy

from dev.pepilica.scripts.tile64_experiment import install

install()
runpy.run_module('dev.pepilica.scripts.benchmark_maps_epoch3', run_name='__main__')
