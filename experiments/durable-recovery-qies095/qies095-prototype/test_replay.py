"""Fixed synthetic traces, fault matrices and bounded resource observations."""
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import (Session, SyntheticEngine, State, Store, MemoryFS, RealFS, Hooks,
                    Invalid, MAX_STEPS, decisions, checkpoint_at)
from recovery import Crash

OUT = Path(sys.argv[1]).resolve()
OUT.mkdir(exist_ok=True)
RESULTS = {}


def fresh():
    fs = MemoryFS()
    Store(fs, 'session095').initialize()
    return fs


def session(fs, hooks=None, engine=None):
    return Session(Store(fs, 'session095'), engine, hooks)


def populate(fs, count, pending=False):
    for _ in range(count):
        s = session(fs)
        s.select_and_prepare()
        s.resume_pending()
    if pending:
        session(fs).select_and_prepare()


def fingerprint(fs):
    return fs.names.copy(), fs.data.copy()


def memory_sample():
    result = {}
    for source in ('status', 'smaps_rollup'):
        for line in Path('/proc/self/' + source).read_text().splitlines():
            key, _, rest = line.partition(':')
            if key in ('VmRSS', 'VmHWM', 'RssAnon', 'Pss'):
                result[key] = int(rest.split()[0]) * 1024
    return result


class ReplayTests(unittest.TestCase):
    def test_01_multitoken_all_restart_points(self):
        fs = fresh()
        populate(fs, 12, pending=True)
        latest = Store(fs, 'session095').recover()
        expected = latest['checkpoint']
        fixture = json.loads((Path(__file__).parent/'fixture.json').read_text())
        self.assertEqual(expected['history'], fixture['evaluated_tokens'])
        self.assertEqual(expected['position'], fixture['position'])
        self.assertEqual(hashlib.sha256(expected['state'].encode()).hexdigest(), fixture['serialized_state_sha256'])
        self.assertEqual(latest['pending'][0]['token'], fixture['pending_token'])
        pending = latest['pending'][0]['token']
        receipts = []
        for start in range(13):
            before = fingerprint(fs)
            s = session(fs)
            got = s.replay(start)
            self.assertEqual(got['state'].history, tuple(expected['history']))
            self.assertEqual(got['state'].blob, expected['state'])
            self.assertEqual(got['state'].position, 12)
            self.assertEqual(got['pending_token'], pending)
            self.assertEqual(got['replayed_tokens'], 12 - start)
            self.assertEqual(fingerprint(fs), before)
            self.assertFalse(got['committed_position_changed'])
            receipts.append({'checkpoint': start, 'replayed': 12-start, 'authoritative_position': 12,
                             'pending_token': pending})
        s = session(fs)
        s.replay(0)
        self.assertEqual(s.resume_pending()['position'], 13)
        count = s.engine.evaluations
        self.assertEqual(s.resume_pending()['position'], 13)
        self.assertEqual(s.engine.evaluations, count)
        self.assertEqual(Store(fs, 'session095').recover()['checkpoint']['history'], expected['history'] + [pending])
        RESULTS['restart_points'] = receipts

    def test_02_interrupted_boundary_matrix(self):
        base = fresh()
        populate(base, 4)
        def cycle(fs, hooks, output):
            s = session(fs, hooks)
            s.select_and_prepare()
            s.attempt_output(lambda data: (output.extend(data), True)[1])
            s.resume_pending()
            s.replay(0)
        normal = copy.deepcopy(base)
        hooks = Hooks(record=True)
        cycle(normal, hooks, bytearray())
        events = hooks.events
        receipts = []
        for cut in range(1, len(events)+1):
            fs = copy.deepcopy(base)
            output = bytearray()
            self.assertRaises(Crash, cycle, fs, Hooks(cut), output)
            for survive in (False, True):
                boot = fs.reboot(survive)
                recovered = Store(boot, 'session095').recover()
                pos = recovered['checkpoint']['position']
                self.assertIn(pos, (4, 5))
                if events[cut-1] in ('after.selection', 'before.journal_sync'):
                    self.assertEqual(pos, 4)
                    self.assertIsNone(recovered['pending'])
                # Default recovery never touches or replays the sink.
                old_output = bytes(output)
                s = session(boot)
                result = s.replay(0)
                self.assertEqual(result['state'].position, pos)
                self.assertEqual(bytes(output), old_output)
                if recovered['pending']:
                    s.resume_pending()
                    self.assertEqual(Store(boot, 'session095').recover()['checkpoint']['position'], 5)
                count = s.engine.evaluations
                s.resume_pending()
                self.assertEqual(s.engine.evaluations, count)
                receipts.append({'cut': cut, 'boundary': events[cut-1], 'survive': survive,
                                 'position': pos, 'pending': bool(recovered['pending']),
                                 'visible_bytes_before_restart': len(output)})
        RESULTS['coordinator_faults'] = receipts

    def test_03_storage_faults_after_multiple_tokens(self):
        base = fresh()
        populate(base, 4)
        trace = copy.deepcopy(base)
        trace.hooks = Hooks(record=True)
        s = session(trace)
        s.select_and_prepare()
        s.resume_pending()
        events = trace.hooks.events
        receipts = []
        for cut in range(1, len(events)+1):
            fs = copy.deepcopy(base)
            fs.hooks = Hooks(cut)
            s = session(fs)
            self.assertRaises(Crash, lambda: (s.select_and_prepare(), s.resume_pending()))
            for survive in (False, True):
                boot = fs.reboot(survive)
                recovered = Store(boot, 'session095').recover()
                self.assertIn(recovered['checkpoint']['position'], (4, 5))
                # Pending cannot be counted as evaluated, including a torn tail.
                expected_pos = recovered['checkpoint']['position']
                replay = session(boot).replay(0)
                self.assertEqual(replay['state'].position, expected_pos)
                if recovered['torn_tail']:
                    self.assertRaises(Invalid, session(boot).resume_pending)
                elif recovered['pending']:
                    self.assertEqual(session(boot).resume_pending()['position'], 5)
                receipts.append({'cut': cut, 'boundary': events[cut-1], 'survive': survive,
                                 'position': expected_pos, 'torn': recovered['torn_tail']})
        RESULTS['multi_token_storage_faults'] = receipts

    def test_04_partial_output(self):
        fs = fresh()
        populate(fs, 3)
        s = session(fs)
        d = s.select_and_prepare()
        visible = bytearray()
        def partial(data):
            visible.extend(data[:1])
            return False
        self.assertRaises(Invalid, s.attempt_output, partial)
        self.assertTrue(s.invalid)
        self.assertEqual(len(visible), 1)
        recovered = session(fs).replay(0)
        self.assertEqual(recovered['authoritative_position'], 3)
        self.assertEqual(recovered['pending_token'], d['token'])
        self.assertEqual(recovered['output']['mode'], 'NO_AUTOMATIC_REPLAY')
        session(fs).resume_pending()
        self.assertEqual(len(visible), 1)
        self.assertEqual(Store(fs, 'session095').recover()['checkpoint']['position'], 4)

    def test_04b_no_implicit_output_retry(self):
        fs = fresh()
        s = session(fs)
        s.select_and_prepare()
        visible = bytearray()
        def sink(data):
            visible.extend(data)
            return True
        result = s.attempt_output(sink)
        self.assertFalse(result['durable_output_ack'])
        before = bytes(visible)
        self.assertRaises(Invalid, s.attempt_output, sink)
        self.assertEqual(bytes(visible), before)
        # New coordinators still report UNKNOWN, never an exactly-once receipt.
        self.assertEqual(session(fs).replay(0)['output']['delivery'], 'UNKNOWN')

    def test_05_engine_drift_and_corruption(self):
        fs = fresh()
        populate(fs, 5)
        old = fingerprint(fs)
        class BadEngine(SyntheticEngine):
            def evaluate(self, state, token, input_position):
                correct = super().evaluate(state, token, input_position)
                return State(correct.position + 1, correct.history, correct.blob)
        self.assertRaises(Invalid, session(fs, engine=BadEngine()).replay, 0)
        self.assertEqual(fingerprint(fs), old)
        session(fs).select_and_prepare()
        old = fingerprint(fs)
        self.assertRaises(Invalid, session(fs, engine=BadEngine()).resume_pending)
        self.assertEqual(fingerprint(fs), old)
        # Corrupt a published checkpoint and a middle journal record separately.
        for name in ('checkpoint-003-000', 'journal'):
            corrupt = copy.deepcopy(fs)
            data = corrupt.read(name, 1 << 20)
            at = len(data)//2
            corrupt.data[corrupt.names[name]] = data[:at] + bytes([data[at] ^ 1]) + data[at+1:]
            self.assertRaises((Invalid, ValueError), session(corrupt).replay, 0)

    def test_06_replay_does_not_reselect(self):
        fs = fresh()
        populate(fs, 8, pending=True)
        class NoSelection(SyntheticEngine):
            def select(self, state):
                raise AssertionError('replay cannot select')
        s = session(fs, engine=NoSelection())
        result = s.replay(3)
        self.assertEqual(result['state'].position, 8)
        self.assertEqual(s.resume_pending()['position'], 9)
        self.assertRaises(Invalid, session(fs).replay, 10)
        self.assertRaises(Invalid, session(fs).replay, 5, 2)

    def test_07_streaming_and_bounds(self):
        fs = fresh()
        populate(fs, MAX_STEPS)
        s = session(fs)
        self.assertEqual(s.replay(0)['state'].position, MAX_STEPS)
        before = fingerprint(fs)
        self.assertRaises(Invalid, s.select_and_prepare)
        self.assertEqual(before, fingerprint(fs))
        generator = decisions(Store(fs, 'session095'), Store(fs, 'session095').recover()['journal_end'])
        self.assertIs(iter(generator), generator)
        self.assertEqual(sum(1 for _ in generator), MAX_STEPS)
        tracemalloc.start()
        for _ in range(100):
            self.assertEqual(session(fs).replay(63)['state'].position, MAX_STEPS)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertLess(peak, 4 * 1024 * 1024)
        self.assertEqual(fs.hooks.events, [])
        RESULTS['bounded_replay'] = {'iterations': 100, 'tokens_in_store': MAX_STEPS,
                                     'tracemalloc_current': current, 'tracemalloc_peak': peak,
                                     'retained_hook_events': len(fs.hooks.events)}

    def test_08_real_restart(self):
        root = OUT/'real-restart'
        root.mkdir()
        fs = RealFS(root)
        try:
            Store(fs, 'session095').initialize()
            populate(fs, 6, pending=True)
        finally:
            fs.close()
        # Reopen filesystem/lock and coordinator, rather than copying live progress.
        fs = RealFS(root)
        try:
            s = session(fs)
            self.assertEqual(s.replay(2)['state'].position, 6)
            self.assertEqual(s.resume_pending()['position'], 7)
            self.assertEqual(s.resume_pending()['position'], 7)
        finally:
            fs.close()

    def test_09_measurements(self):
        rows = []
        for count in (1, 4, 8, 16, 32, 64):
            root = OUT/('benchmark-%02d' % count)
            root.mkdir()
            fs = RealFS(root)
            try:
                Store(fs, 'session095').initialize()
                populate(fs, count)
                sizes = {p.name:p.stat().st_size for p in root.iterdir()}
                memory_before = memory_sample()
                tracemalloc.start()
                times = []
                memory_during = []
                for _ in range(3):
                    s = session(fs)
                    start = time.perf_counter_ns()
                    replay = s.replay(0)
                    times.append((time.perf_counter_ns()-start)/1e6)
                    self.assertEqual(replay['state'].position, count)
                    self.assertEqual(s.engine.evaluations, count)
                    memory_during.append(memory_sample())
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                self.assertLess(peak, 4*1024*1024)
                rows.append({'tokens':count, 'journal_bytes':sizes['journal'],
                             'checkpoint_bytes':sum(v for k,v in sizes.items() if k.startswith('checkpoint-')),
                             'receipt_bytes':sum(v for k,v in sizes.items() if k.startswith('commit-')),
                             'total_logical_bytes':sum(sizes.values()),
                             'total_allocated_bytes':sum(p.stat().st_blocks*512 for p in root.iterdir()),
                             'replay_times_ms':times, 'replay_median_ms':statistics.median(times),
                             'tracemalloc_current':current, 'tracemalloc_peak':peak,
                             'memory_before':memory_before, 'memory_observations':memory_during})
            finally:
                fs.close()
        RESULTS['measurements'] = rows


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReplayTests))
    (OUT/'results.json').write_text(json.dumps(RESULTS,indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
