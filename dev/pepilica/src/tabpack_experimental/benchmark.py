"""Trace summarization hook used by common.profiling; no training runner."""

from collections import Counter, defaultdict

PHASES = ('forward', 'loss', 'zero_grad', 'backward', 'optimizer')


def _activity_union_us(events):
    intervals = sorted((e['ts'], e['ts'] + e['dur']) for e in events)
    end, total = float('-inf'), 0.0
    for left, right in intervals:
        total += max(0, right - max(left, end))
        end = max(end, right)
    return total


def summarize_trace(trace, steps, cuda, phase_names=PHASES):
    events = [e for e in trace['traceEvents'] if e.get('ph') == 'X']
    ranges = sorted(
        (e['ts'], e['ts'] + e['dur'], e['name'].split('/')[-1])
        for e in events
        if e.get('cat') == 'user_annotation'
        and e.get('name') in {f'tabpack/{p}' for p in phase_names}
    )

    def phase_at(event):
        ts = event['ts']
        for left, right, phase in ranges:
            if left <= ts < right:
                return phase
        return 'unassigned'

    cpu_ops = [e for e in events if e.get('cat') == 'cpu_op']
    owners = {
        e['args']['External id']: e
        for e in cpu_ops
        if 'External id' in e.get('args', {})
    }
    runtime = {
        e['args']['correlation']: e
        for e in events
        if e.get('cat') in ('cuda_runtime', 'cuda_driver')
        and 'correlation' in e.get('args', {})
    }
    operators = defaultdict(lambda: {'calls': 0, 'kernels': 0, 'kernel_us': 0.0})
    phases = {
        p: {'aten_calls': 0, 'kernels': 0, 'kernel_us': 0.0}
        for p in (*phase_names, 'unassigned')
    }
    for e in cpu_ops:
        if e['name'].startswith('aten::'):
            phase = phase_at(e)
            operators[(phase, e['name'])]['calls'] += 1
            phases[phase]['aten_calls'] += 1
    kernels = [e for e in events if e.get('cat') == 'kernel']
    kernel_names = defaultdict(lambda: {'calls': 0, 'duration_us': 0.0})
    for e in kernels:
        args = e.get('args', {})
        owner = owners.get(args.get('External id'))
        launch = runtime.get(args.get('correlation'))
        phase = (
            phase_at(owner if owner is not None else launch)
            if (owner is not None or launch is not None)
            else 'unassigned'
        )
        phases[phase]['kernels'] += 1
        phases[phase]['kernel_us'] += e['dur']
        name = owner['name'] if owner is not None else '<unknown>'
        operators[(phase, name)]['kernels'] += 1
        operators[(phase, name)]['kernel_us'] += e['dur']
        kernel_names[e['name']]['calls'] += 1
        kernel_names[e['name']]['duration_us'] += e['dur']
    activities = [
        e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')
    ]
    busy = _activity_union_us(activities)
    span = (
        (max(e['ts'] + e['dur'] for e in activities) - min(e['ts'] for e in activities))
        if activities
        else 0
    )
    allocations = [
        e
        for e in trace['traceEvents']
        if e.get('name') == '[memory]' and e.get('args', {}).get('Bytes', 0) > 0
    ]
    return {
        'steps': steps,
        'cuda_kernels_available': bool(kernels) if cuda else None,
        'phases': {
            p: {
                'aten_calls_per_step': v['aten_calls'] / steps,
                'kernel_launches_per_step': v['kernels'] / steps if cuda else None,
                'kernel_time_ms_per_step': v['kernel_us'] / 1000 / steps
                if cuda
                else None,
            }
            for p, v in phases.items()
        },
        'operators': [
            {
                'phase': p,
                'name': name,
                'calls_per_step': v['calls'] / steps,
                'kernel_launches_per_step': v['kernels'] / steps if cuda else None,
                'kernel_time_ms_per_step': v['kernel_us'] / 1000 / steps
                if cuda
                else None,
            }
            for (p, name), v in sorted(
                operators.items(), key=lambda item: -item[1]['kernel_us']
            )
        ],
        'kernels': [
            {
                'name': name,
                'launches_per_step': v['calls'] / steps,
                'time_ms_per_step': v['duration_us'] / 1000 / steps,
            }
            for name, v in sorted(
                kernel_names.items(), key=lambda item: -item[1]['duration_us']
            )
        ],
        'gpu_activity': {
            'union_ms': busy / 1000,
            'envelope_ms': span / 1000,
            'gap_fraction_within_envelope': 1 - busy / span if span else None,
            'memcpy_events': Counter(e['cat'] for e in activities)['gpu_memcpy'],
            'memset_events': Counter(e['cat'] for e in activities)['gpu_memset'],
        }
        if cuda
        else None,
        'allocation_requests': {
            'count': len(allocations),
            'bytes': sum(e['args']['Bytes'] for e in allocations),
        },
        'notes': [
            (
                'Kernel phases use CPU External id or runtime/driver launch '
                'correlation; '
                'asynchronous GPU timestamps are never matched to CPU phase ranges.'
            ),
            (
                'ATen calls include nested operators; kernel attribution counts each '
                'launch once, including launches on autograd worker threads.'
            ),
            'GPU gaps are from an instrumented trace, not hardware utilization.',
            'Allocation requests are tensor allocations, not cudaMalloc calls.',
        ],
    }
