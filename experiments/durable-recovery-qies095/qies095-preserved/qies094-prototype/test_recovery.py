"""Deterministic model faults plus real filesystem/SIGKILL tests; no model loading."""
import copy
import errno
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time
import tracemalloc
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recovery import (Store, MemoryFS, RealFS, Hooks, Crash, Invalid, encode, decode,
                      digest, MAX_STEPS, HEADER, MAGIC, VERSION)

OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path('qies094-checks').resolve()
OUT.mkdir(exist_ok=True)
RESULTS = {}


def new_memory():
    fs = MemoryFS()
    Store(fs, 'session094').initialize()
    return fs


def workflow(fs):
    store = Store(fs, 'session094')
    store.prepare(19, b'4')
    store.evaluate_pending()


def mutate(fs, name, data):
    fs.data[fs.names[name]] = data


def usage():
    values = {}
    for name in ('status', 'smaps_rollup'):
        for line in Path('/proc/self/' + name).read_text().splitlines():
            key, _, rest = line.partition(':')
            if key in ('VmRSS', 'VmHWM', 'RssAnon', 'Pss', 'Private_Dirty'):
                values[key] = int(rest.split()[0]) * 1024
    mem = Path('/proc/meminfo').read_text().splitlines()
    values['MemAvailable'] = int(next(l.split()[1] for l in mem if l.startswith('MemAvailable:'))) * 1024
    try:
        rel = next(l.split(':', 2)[2].strip() for l in Path('/proc/self/cgroup').read_text().splitlines() if l.startswith('0:'))
        cg = Path('/sys/fs/cgroup') / rel.lstrip('/')
        for name in ('memory.current', 'memory.peak', 'memory.swap.current'):
            values['cgroup_' + name] = int((cg / name).read_text())
    except (OSError, StopIteration, ValueError):
        values['cgroup_accounting'] = 'unavailable'
    return values


class RecoveryTests(unittest.TestCase):
    def test_01_pending_and_continuation(self):
        fs = new_memory()
        for token in (19, 7, 3):
            s = Store(fs, 'session094')
            before = s.recover()['checkpoint']['position']
            s.prepare(token, b'piece')
            state = s.recover()
            self.assertEqual(state['checkpoint']['position'], before)
            self.assertEqual(state['pending'][0]['token'], token)
            self.assertIn('UNKNOWN', state['output_delivery'])
            Store(fs, 'session094').evaluate_pending()
            state = Store(fs, 'session094').recover()
            self.assertEqual(state['checkpoint']['position'], before + 1)
            self.assertIsNone(state['pending'])
        self.assertEqual(state['checkpoint']['history'], [19, 7, 3])
        self.assertIn('checkpoint-000-000', fs.names)
        self.assertIn('checkpoint-002-000', fs.names)
        self.assertRaises(Invalid, Store(fs, 'session094').evaluate_pending)

    def test_02_every_crash_boundary(self):
        base = new_memory()
        normal = copy.deepcopy(base)
        normal.hooks = Hooks(record=True)
        workflow(normal)
        events = normal.hooks.events
        self.assertIn('mid.write:journal', events)
        self.assertIn('before.dirsync', events)
        receipts = []
        for cut in range(1, len(events) + 1):
            fs = copy.deepcopy(base)
            fs.hooks = Hooks(cut)
            with self.assertRaises(Crash):
                workflow(fs)
            for retain in (False, True):
                boot = fs.reboot(retain)
                state = Store(boot, 'session094').recover()
                pos = state['checkpoint']['position']
                self.assertIn(pos, (0, 1))
                self.assertEqual(state['checkpoint']['history'], [19] if pos else [])
                if pos:
                    self.assertIsNone(state['pending'])
                if events[cut-1] == 'coordinator.committed':
                    self.assertEqual(pos, 1)
                # A surviving complete unsynced decision may be adopted on reboot.
                # A torn tail is read-only and refuses any new append.
                if state['torn_tail']:
                    self.assertRaises(Invalid, Store(boot, 'session094').prepare, 1, b'x')
                elif state['pending']:
                    Store(boot, 'session094').evaluate_pending()
                    self.assertEqual(Store(boot, 'session094').recover()['checkpoint']['position'], 1)
                receipts.append({'cut': cut, 'boundary': events[cut-1], 'retain_unsynced': retain,
                                 'position': pos, 'pending': bool(state['pending']), 'torn': state['torn_tail']})
        RESULTS['faults'] = {'boundaries': len(events), 'outcomes': len(receipts), 'receipts': receipts}

    def test_03_recovery_scan_interruption(self):
        base = new_memory()
        workflow(base)
        base.hooks = Hooks(record=True)
        Store(base, 'session094').recover()
        events = base.hooks.events[:]
        for cut in range(1, len(events) + 1):
            fs = copy.deepcopy(base)
            old_names, old_data = fs.names.copy(), fs.data.copy()
            fs.hooks = Hooks(cut)
            self.assertRaises(Crash, Store(fs, 'session094').recover)
            self.assertEqual(fs.names, old_names)
            self.assertEqual(fs.data, old_data)
            self.assertEqual(Store(fs.reboot(), 'session094').recover()['checkpoint']['position'], 1)
        RESULTS['recovery_scan_faults'] = len(events)

    def test_04_corruption_rejected(self):
        base = new_memory()
        workflow(base)
        for name in ('identity', 'journal', 'checkpoint-001-000', 'commit-001'):
            fs = copy.deepcopy(base)
            data = fs.read(name, 1 << 20)
            mutate(fs, name, data[:-1] + bytes([data[-1] ^ 1]))
            self.assertRaises((Invalid, ValueError), Store(fs, 'session094').recover)
        self.assertRaises(Invalid, Store(base, 'otherSession').recover)
        for field, value in (('position', True), ('history', [7]), ('state', 'corrupt'), ('identity', 'other')):
            fs = copy.deepcopy(base)
            cp = decode(fs.read('checkpoint-001-000'))
            cp[field] = value
            blob = encode(cp)
            mutate(fs, 'checkpoint-001-000', blob)
            receipt = decode(fs.read('commit-001'))
            receipt['checkpoint'] = digest(blob)
            mutate(fs, 'commit-001', encode(receipt))
            self.assertRaises(Invalid, Store(fs, 'session094').recover)
        for bad in (b'{"schema":1,"schema":1}', b'{ "x": 1 }'):
            self.assertRaises(Invalid, decode, bad)
        fs = copy.deepcopy(base)
        receipt = decode(fs.read('commit-001'))
        receipt['file'] = '../escape'
        mutate(fs, 'commit-001', encode(receipt))
        self.assertRaises(Invalid, Store(fs, 'session094').recover)
        # Complete trailing record with a bad checksum is corruption, not a torn tail.
        fs = copy.deepcopy(base)
        Store(fs, 'session094').prepare(7, b'b')
        raw = fs.read('journal', 1 << 20)
        mutate(fs, 'journal', raw[:-1] + bytes([raw[-1] ^ 1]))
        self.assertRaises(Invalid, Store(fs, 'session094').recover)
        # Wrong frame version/oversized length also fail closed.
        fs = new_memory()
        mutate(fs, 'journal', HEADER.pack(MAGIC, VERSION + 1, 0))
        self.assertRaises(Invalid, Store(fs, 'session094').recover)

    def test_05_torn_tail_refusal(self):
        fs = new_memory()
        fs.write('journal', b'QIE', append=True)
        s = Store(fs, 'session094')
        self.assertTrue(s.recover()['torn_tail'])
        before = fs.read('journal')
        self.assertRaises(Invalid, s.prepare, 19, b'4')
        self.assertEqual(fs.read('journal'), before)

    def test_06_capacity(self):
        fs = new_memory()
        for i in range(MAX_STEPS):
            s = Store(fs, 'session094')
            s.prepare(i, b'x')
            s.evaluate_pending()
        s = Store(fs, 'session094')
        self.assertEqual(s.recover()['checkpoint']['position'], MAX_STEPS)
        names = fs.names.copy()
        self.assertRaises(Invalid, s.prepare, 1, b'x')
        self.assertEqual(names, fs.names)
        RESULTS['capacity'] = {'max_tokens': MAX_STEPS, 'files': len(fs.names),
                               'logical_bytes': sum(len(v) for v in fs.data.values())}

    def test_07_io_errors(self):
        base = new_memory()
        trace = copy.deepcopy(base)
        trace.hooks = Hooks(record=True)
        workflow(trace)
        points = [(i, label) for i, label in enumerate(trace.hooks.events, 1)
                  if label.startswith(('before.write', 'mid.write', 'after.write', 'before.fsync', 'before.dirsync'))]
        count = 0
        for number in (errno.ENOSPC, errno.EIO):
            for cut, label in points:
                fs = copy.deepcopy(base)
                def fail():
                    raise OSError(number, 'injected')
                fs.hooks = Hooks(cut, fail)
                s = Store(fs, 'session094')
                self.assertRaises(OSError, lambda: (s.prepare(19, b'4'), s.evaluate_pending()))
                self.assertTrue(s.poisoned)
                self.assertRaises(Invalid, s.prepare, 7, b'b')
                recovered = Store(fs.reboot(), 'session094').recover()
                self.assertIn(recovered['checkpoint']['position'], (0, 1))
                count += 1
        RESULTS['io_errors'] = count

    def test_08_process_termination(self):
        probe = new_memory()
        probe.hooks = Hooks(record=True)
        workflow(probe)
        events = probe.hooks.events
        roots = OUT / 'killed-processes'
        roots.mkdir()
        receipts = []
        for cut, label in enumerate(events, 1):
            # Kill at every mutating/commit boundary; scan faults covered above.
            if label.startswith('scan.'):
                continue
            root = roots / ('cut-%03d' % cut)
            root.mkdir()
            fs = RealFS(root)
            Store(fs, 'session094').initialize()
            fs.close()
            pid = os.fork()
            if pid == 0:
                def die():
                    os.kill(os.getpid(), signal.SIGKILL)
                fs = RealFS(root, Hooks(cut, die))
                workflow(fs)
                os._exit(97)
            got, status = os.waitpid(pid, 0)
            self.assertEqual(got, pid)
            self.assertTrue(os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL, (cut, label, status))
            fs = RealFS(root)
            try:
                state = Store(fs, 'session094').recover()
                self.assertIn(state['checkpoint']['position'], (0, 1))
                receipts.append({'cut': cut, 'boundary': label, 'position': state['checkpoint']['position'],
                                 'pending': bool(state['pending']), 'torn': state['torn_tail']})
            finally:
                fs.close()
        RESULTS['sigkill'] = receipts

    def test_09_real_lock_and_writes(self):
        root = OUT / 'real-write-tests'
        root.mkdir()
        fs = RealFS(root)
        try:
            self.assertRaises(BlockingIOError, RealFS, root)
            Store(fs, 'session094').initialize()
            original = os.write
            calls = [0]
            def short_write(fd, data):
                calls[0] += 1
                if calls[0] % 7 == 0:
                    raise InterruptedError()
                return original(fd, data[:3])
            os.write = short_write
            try:
                workflow(fs)
            finally:
                os.write = original
            self.assertGreater(calls[0], 7)
            self.assertEqual(Store(fs, 'session094').recover()['checkpoint']['position'], 1)
            s = Store(fs, 'session094')
            os.write = lambda fd, data: 0
            try:
                self.assertRaises(OSError, s.prepare, 7, b'b')
            finally:
                os.write = original
            self.assertTrue(s.poisoned)
        finally:
            fs.close()

    def test_10_initialization_faults(self):
        trace = MemoryFS(Hooks(record=True))
        Store(trace, 'session094').initialize()
        count = 0
        for cut in range(1, len(trace.hooks.events) + 1):
            fs = MemoryFS(Hooks(cut))
            self.assertRaises(Crash, Store(fs, 'session094').initialize)
            for retain in (False, True):
                boot = fs.reboot(retain)
                try:
                    state = Store(boot, 'session094').recover()
                except (Invalid, KeyError):
                    pass
                else:
                    self.assertEqual(state['checkpoint']['position'], 0)
                count += 1
        RESULTS['initialization_faults'] = count

    def test_10b_mixed_persistence(self):
        base = new_memory()
        trace = copy.deepcopy(base)
        trace.hooks = Hooks(record=True)
        workflow(trace)
        total = 0
        rejected = 0
        for cut in range(1, len(trace.hooks.events) + 1):
            fs = copy.deepcopy(base)
            fs.hooks = Hooks(cut)
            self.assertRaises(Crash, workflow, fs)
            for names_survive, data_survive in ((False, True), (True, False)):
                boot = fs.reboot(names_survive=names_survive, data_survive=data_survive)
                try:
                    result = Store(boot, 'session094').recover()
                except Invalid:
                    rejected += 1
                else:
                    pos = result['checkpoint']['position']
                    self.assertIn(pos, (0, 1))
                    self.assertEqual(result['checkpoint']['history'], [19] if pos else [])
                total += 1
        RESULTS['mixed_persistence'] = {'outcomes': total, 'explicit_rejections': rejected}

    def test_10c_bounded_observers(self):
        fs = new_memory()
        workflow(fs)
        fs.hooks = Hooks()
        tracemalloc.start()
        for _ in range(1000):
            Store(fs, 'session094').recover()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertEqual(fs.hooks.events, [])
        self.assertLess(peak, 4 * 1024 * 1024)
        RESULTS['repeated_recovery_memory'] = {'iterations': 1000, 'current': current, 'peak': peak,
                                              'recorded_hook_events': len(fs.hooks.events)}

    def test_11_benchmark(self):
        root = OUT / 'benchmark'
        root.mkdir()
        fs = RealFS(root)
        try:
            store = Store(fs, 'session094')
            store.initialize()
            store.recover()
            before = usage()
            tracemalloc.start()
            tracemalloc.reset_peak()
            start_alloc, _ = tracemalloc.get_traced_memory()
            journal, checkpoint, scanning, memory = [], [], [], []
            journal_io, checkpoint_io = [], []
            growth = []
            previous = sum(p.stat().st_size for p in root.iterdir())
            for i in range(32):
                start = time.perf_counter_ns()
                store.prepare(i, b'x')
                journal.append((time.perf_counter_ns() - start) / 1e6)
                journal_io.append(store.last_journal_ns / 1e6)
                start = time.perf_counter_ns()
                store.evaluate_pending()
                checkpoint.append((time.perf_counter_ns() - start) / 1e6)
                checkpoint_io.append(store.last_checkpoint_ns / 1e6)
                start = time.perf_counter_ns()
                state = store.recover()
                scanning.append((time.perf_counter_ns() - start) / 1e6)
                self.assertEqual(state['checkpoint']['position'], i + 1)
                logical = sum(p.stat().st_size for p in root.iterdir())
                growth.append(logical - previous)
                previous = logical
                memory.append(usage())
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            after = usage()
            def stats(values):
                return {'min_ms': min(values), 'median_ms': statistics.median(values), 'max_ms': max(values)}
            RESULTS['benchmark'] = {'tokens': 32, 'journal_prepare_including_recovery': stats(journal),
                'checkpoint_publish_including_recovery': stats(checkpoint), 'recovery_scan': stats(scanning),
                'journal_write_and_sync': stats(journal_io), 'checkpoint_publish_only': stats(checkpoint_io),
                'memory_before': before, 'memory_after': after,
                'max_observed_RssAnon': max(x['RssAnon'] for x in memory),
                'max_observed_Pss': max(x['Pss'] for x in memory),
                'tracemalloc_incremental_current': current - start_alloc, 'tracemalloc_peak': peak,
                'logical_growth_per_token': growth,
                'logical_total': previous, 'allocated_total': sum(p.stat().st_blocks * 512 for p in root.iterdir()),
                'retained_files': len(list(root.iterdir()))}
            self.assertLess(peak, 4 * 1024 * 1024)
            self.assertLess(max(x['RssAnon'] for x in memory) - before['RssAnon'], 4 * 1024 * 1024)
        finally:
            fs.close()


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests))
    (OUT / 'results.json').write_text(json.dumps(RESULTS, indent=2) + '\n')
    sys.exit(not result.wasSuccessful())
