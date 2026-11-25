"""Run ActiveFixedSlotMLP by default; delegate training/profiling to common.run."""

import argparse
import copy
import json
import sys
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest.mock import patch

import tabpackage._training as training
from common import run
from dev.pepilica.scripts import DEFAULT_VARIANT
from dev.pepilica.scripts.epoch_windows import CandidateTracker, EpochWindows


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--measure-epochs', type=int, nargs='+')
    args, _ = parser.parse_known_args(argv)
    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument('--measure-epochs', type=int, nargs='+')
    custom.add_argument('--variant', default=DEFAULT_VARIANT)
    selected, common_argv = custom.parse_known_args(argv)
    common_argv.extend(['--variant', selected.variant])
    if args.measure_epochs and (
        min(args.measure_epochs) < 1
        or args.measure_epochs != sorted(set(args.measure_epochs))
    ):
        parser.error('--measure-epochs must be positive, strictly increasing epochs')
    backends = []
    tracker = CandidateTracker()
    windows = []
    original_factory, original_remove = (
        training.make_optimizer_pack,
        training.pack_remove,
    )
    original_variants = run.VARIANTS.copy()

    def record_hyperparameters(kwargs):
        tracker.initialize(kwargs)
        if args.out is not None:
            (args.out / 'sampled-hyperparameters.json').write_text(
                json.dumps(copy.deepcopy(kwargs['per_model_hyperparameters']), indent=2)
                + '\n'
            )
        return kwargs

    with ExitStack() as patches:

        def remove_members(*values, pack_idx, **options):
            tracker.remove(pack_idx)
            result = original_remove(*values, pack_idx=pack_idx, **options)
            return result

        patches.enter_context(patch.object(training, 'pack_remove', remove_members))
        if args.measure_epochs:
            original_load = run.data.load

            def load(cfg):
                arrays, meta = original_load(cfg)
                tracker.steps_per_epoch = (
                    len(arrays['train']['y']) + cfg['batch_size'] - 1
                ) // cfg['batch_size']
                return arrays, meta

            def instrumenter(mode, out, schedule, **options):
                steps = tracker.steps_per_epoch
                if schedule.start_step != (args.measure_epochs[0] - 1) * steps:
                    raise ValueError(
                        '--epoch must equal the first --measure-epochs entry'
                    )
                if schedule.total and schedule.total > steps:
                    raise ValueError(
                        'Each measurement window must fit inside its epoch'
                    )
                obj = EpochWindows(
                    mode,
                    out,
                    schedule,
                    epochs=args.measure_epochs,
                    steps_per_epoch=steps,
                    tracker=tracker,
                    **options,
                )
                windows.append(obj)
                return obj

            patches.enter_context(patch.object(run.data, 'load', load))
            patches.enter_context(patch.object(run, 'Instrumenter', instrumenter))

        def native(kwargs):
            return original_variants['native'](record_hyperparameters(kwargs))

        def active(kwargs, *, grouped_first):
            if any(kwargs.get(key, False) for key in run.COMPILE_KEYS):
                raise ValueError(
                    'This comparison requires all torch.compile flags disabled'
                )
            from tabpack_experimental.tabpack_adapter import ActiveTabPackBackend

            hyperparameters = kwargs['per_model_hyperparameters']
            record_hyperparameters(kwargs)
            ids = tracker.ids

            def make_optimizer(model, **options):
                optimizer = original_factory(model, **options)
                backends.append(
                    ActiveTabPackBackend(
                        model,
                        optimizer,
                        ids,
                        hyperparameters,
                        grouped_first=grouped_first,
                    )
                )
                return optimizer

            patches.enter_context(
                patch.object(training, 'make_optimizer_pack', make_optimizer)
            )
            return kwargs

        run.VARIANTS.clear()
        run.VARIANTS.update(
            native=native,
            # Native first block/head; grouped square hidden matrices.
            active_square_matrices_only=partial(active, grouped_first=False),
            active_grouped_first=partial(active, grouped_first=True),
        )
        try:
            report = run.main(common_argv)
            if args.out is not None:
                metadata = {
                    'runner': 'common.run',
                    'layout': 'active-only activations; fixed parameter slots'
                    if backends
                    else 'native padded TabPack',
                    'grouped_first': selected.variant == 'active_grouped_first',
                    'grouped_output_head': False,
                    'hidden_tiles': [64, 64] if backends else None,
                    'torch_compile': False,
                    'cuda_graph_replay': False,
                    'routing_changes': tracker.routes,
                    'remaining_candidates': tracker.snapshot(),
                    'backends': [
                        {
                            'type': type(backend).__name__,
                            'remaining_ids': backend.ids.copy(),
                            'widths': backend.widths,
                            'hidden_depths': backend.depths,
                            'forward_calls': backend.calls,
                            'fallback_calls': backend.fallback_calls,
                            'transfer_rebuilds': backend.rebuilds,
                        }
                        for backend in backends
                    ],
                }
                (args.out / 'backend.json').write_text(
                    json.dumps(metadata, indent=2) + '\n'
                )
            return report
        finally:
            for window in windows:
                window.close()
            for backend in reversed(backends):
                backend.remove()
            run.VARIANTS.clear()
            run.VARIANTS.update(original_variants)


if __name__ == '__main__':
    main()
