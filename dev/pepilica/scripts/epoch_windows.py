"""Select several epochs while delegating each measurement to common.profiling."""

import contextlib
import dataclasses
import json
from pathlib import Path

from common.profiling import Instrumenter, Schedule


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


class CandidateTracker:
    def __init__(self):
        self.ids = []
        self.hyperparameters = []
        self.routes = []
        self.steps_per_epoch = None
        self.clock = None

    def initialize(self, kwargs):
        self.ids[:] = range(kwargs['n_models'])
        self.hyperparameters = kwargs['per_model_hyperparameters']

    def remove(self, pack_idx):
        removed = set(pack_idx.tolist())
        before = self.ids.copy()
        self.ids[:] = [i for local, i in enumerate(self.ids) if local not in removed]
        step = (
            self.clock.step
            if self.clock is not None and self.clock.mode != 'none'
            else None
        )
        self.routes.append(
            {
                'before': before,
                'after': self.ids.copy(),
                'step': step,
                'epoch': (
                    (step + self.steps_per_epoch - 1) // self.steps_per_epoch
                    if step is not None
                    else None
                ),
            }
        )

    def snapshot(self):
        widths = [self.hyperparameters[i]['model']['d_block'] for i in self.ids]
        return {
            'count': len(self.ids),
            'ids': self.ids.copy(),
            'widths': widths,
            'sum_hidden_widths': sum(widths),
        }


class EpochWindows:
    """Use a fresh common Instrumenter per requested epoch of a single training run.

    No timing/profiler/memory measurement is performed between these windows.
    Finished windows are summarized immediately and their instrumenters released.
    The broad schedule only tells common.run how far this training must continue.
    """

    def __init__(
        self,
        mode,
        out,
        schedule,
        *,
        epochs,
        steps_per_epoch,
        tracker,
        capture_phase=None,
        torch_options=None,
    ):
        self.epochs = tuple(epochs)
        self.mode, self.out = mode, Path(out)
        self.steps_per_epoch = steps_per_epoch
        self.tracker = tracker
        self.window = schedule
        self.capture_phase, self.torch_options = capture_phase, torch_options
        self.step = 0
        self.index = 0
        self.child = None
        self.results = {}
        self.pack_start = None
        start = (self.epochs[0] - 1) * steps_per_epoch
        end = (self.epochs[-1] - 1) * steps_per_epoch + (
            schedule.total or steps_per_epoch
        )
        self.schedule = Schedule(start, end - start)
        tracker.clock = self

    @property
    def done(self):
        return len(self.results) == len(self.epochs)

    def phase(self, name):
        if self.done or self.mode == 'none':
            return contextlib.nullcontext()
        epoch = self.epochs[self.index]
        start = (epoch - 1) * self.steps_per_epoch
        if self.child is None:
            if self.step != start or name != 'gather':
                return contextlib.nullcontext()
            self.pack_start = self.tracker.snapshot()
            self.child = Instrumenter(
                self.mode,
                self.out / f'epoch-{epoch}',
                dataclasses.replace(self.window, start_step=start),
                capture_phase=self.capture_phase,
                torch_options=self.torch_options,
            )
            self.child.step = self.step
        return self.child.phase(name)

    def step_end(self):
        if self.child is not None:
            self.child.step_end()
        self.step += 1
        self._finish_if_done()

    def epoch_end(self):
        if self.child is not None:
            self.child.epoch_end()
        self._finish_if_done()

    def _finish_if_done(self):
        if self.child is None or not self.child.done:
            return
        epoch = self.epochs[self.index]
        result = {
            'epoch': epoch,
            'completed': True,
            'observed_training_steps': self.child.step - self.child.start_step,
            'pack_start': self.pack_start,
            'pack_end': self.tracker.snapshot(),
            'train_profile': self.child.summary(),
        }
        self.results[str(epoch)] = result
        write(self.out / f'epoch-{epoch}' / 'report.json', result)
        self.child = None
        self.pack_start = None
        self.index += 1

    def summary(self):
        results = dict(self.results)
        for epoch in self.epochs:
            if str(epoch) not in results:
                results[str(epoch)] = {
                    'epoch': epoch,
                    'completed': False,
                    'reason': (
                        'unprofiled run'
                        if self.mode == 'none'
                        else 'training ended before this measurement window completed'
                    ),
                }
        return {
            'mode': self.mode,
            'requested_epochs': list(self.epochs),
            'epochs': results,
            'steps_per_epoch': self.steps_per_epoch,
            'schedule': dataclasses.asdict(self.schedule),
        }

    def close(self):
        if self.child is not None and self.child.active:
            self.child._end()
            self.child = None
