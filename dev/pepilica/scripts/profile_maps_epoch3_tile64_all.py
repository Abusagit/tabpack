"""Profile Maps Routing with 64x64 fixed-slot and Muon GEMM tiles."""
import runpy

from dev.pepilica.scripts.tile64_experiment import install

install()
runpy.run_module('dev.pepilica.scripts.profile_maps_epoch3', run_name='__main__')
