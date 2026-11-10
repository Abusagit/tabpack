import runpy

from dev.pepilica.scripts.tile64_experiment import install

install()
runpy.run_module('dev.pepilica.scripts.profile_homecredit_epoch3', run_name='__main__')
