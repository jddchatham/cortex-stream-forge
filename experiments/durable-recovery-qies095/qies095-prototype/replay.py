"""Synthetic recovery coordinator. No model/native engine imports or automatic output replay."""
import base64
from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'qies095-preserved/qies094-prototype'))
from recovery import (Store, RealFS, MemoryFS, Hooks, Invalid, HEADER, MAGIC, VERSION,
                      MAX_STEPS, ZERO, decode, encode, synthetic, integer)


@dataclass(frozen=True)
class State:
    position: int
    history: tuple
    blob: str


class SyntheticEngine:
    """Stateless fixture transition; no persistent native context or output ownership."""
    def __init__(self):
        self.evaluations = 0

    def restore(self, checkpoint):
        state = State(checkpoint['position'], tuple(checkpoint['history']), checkpoint['state'])
        self.verify(state)
        return state

    def verify(self, state):
        if state.position != len(state.history) or state.blob != synthetic(list(state.history)):
            raise Invalid('synthetic state identity/position')

    def evaluate(self, state, token, input_position):
        self.verify(state)
        integer(token, 255)
        if state.position != input_position:
            raise Invalid('replay position mismatch')
        self.evaluations += 1
        history = state.history + (token,)
        return State(input_position + 1, history, synthetic(list(history)))

    def select(self, state):
        self.verify(state)
        # Deliberately varies across positions and differs from a constant token.
        return (sum(state.history) * 17 + state.position * 3 + 19) % 251


def checkpoint_at(store, position):
    receipt = decode(store.fs.read('commit-%03d' % position))
    return decode(store.fs.read(receipt['file']))


def decisions(store, boundary):
    """Stream one bounded frame at a time after Store.recover validates the snapshot."""
    fs = store.fs
    if isinstance(fs, RealFS):
        fd = os.open('journal', os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fs.dirfd)
        stream = os.fdopen(fd, 'rb')
    else:
        stream = io.BytesIO(fs.read('journal', (MAX_STEPS + 1) * 8192))
    previous = ZERO
    offset = 0
    count = 0
    try:
        while offset < boundary:
            header = stream.read(HEADER.size)
            if len(header) != HEADER.size:
                raise Invalid('journal changed during replay')
            magic, version, size = HEADER.unpack(header)
            if magic != MAGIC or version != VERSION or size > 8192:
                raise Invalid('replay frame bounds')
            payload = stream.read(size)
            checksum = stream.read(32)
            frame_hash = hashlib.sha256(header + payload).hexdigest()
            if len(payload) != size or len(checksum) != 32 or checksum.hex() != frame_hash:
                raise Invalid('replay checksum')
            value = decode(payload)
            count += 1
            if count > MAX_STEPS or value['sequence'] != count or value['previous'] != previous:
                raise Invalid('replay sequence')
            offset += HEADER.size + size + 32
            if offset > boundary:
                raise Invalid('invalid prefix boundary')
            previous = frame_hash
            yield value
        if offset != boundary:
            raise Invalid('prefix mismatch')
    finally:
        stream.close()


class Session:
    def __init__(self, store, engine=None, hooks=None):
        self.store = store
        self.engine = engine or SyntheticEngine()
        self.hooks = hooks or Hooks()
        self.invalid = False
        self.output_sequence_attempted = 0

    def _valid(self):
        if self.invalid:
            raise Invalid('failed session; restore through new coordinator')

    def select_and_prepare(self):
        self._valid()
        try:
            recovered = self.store.recover()
            if recovered['pending'] or recovered['torn_tail']:
                raise Invalid('cannot select over pending/torn journal')
            current = self.engine.restore(recovered['checkpoint'])
            self.hooks.hit('before.selection')
            token = self.engine.select(current)
            integer(token, 255)
            self.hooks.hit('after.selection')
            self.hooks.hit('before.journal_sync')
            decision = self.store.prepare(token, ('t%03d;' % token).encode('ascii'))
            self.hooks.hit('after.journal_sync')
            return decision
        except BaseException:
            self.invalid = True
            raise

    def attempt_output(self, sink):
        """Explicit live attempt only. No durable delivery acknowledgment is implied."""
        self._valid()
        try:
            recovered = self.store.recover()
            if recovered['torn_tail'] or not recovered['pending']:
                raise Invalid('pending output decision required')
            decision = recovered['pending'][0]
            if decision['sequence'] <= self.output_sequence_attempted:
                raise Invalid('output already attempted by this coordinator')
            self.output_sequence_attempted = decision['sequence']
            self.hooks.hit('before.output')
            data = base64.b64decode(decision['piece'], validate=True)
            if sink(data) is not True:
                raise Invalid('output failure; delivery may be partial')
            self.hooks.hit('after.output')
            return {'decision_sequence': decision['sequence'], 'local_write_returned': True,
                    'external_delivery': 'UNKNOWN', 'durable_output_ack': False}
        except BaseException:
            self.invalid = True
            raise

    def resume_pending(self):
        """Evaluate at the original schedule boundary; never emit or select here."""
        self._valid()
        try:
            recovered = self.store.recover()
            if recovered['torn_tail']:
                raise Invalid('torn journal is read-only')
            if not recovered['pending']:
                return recovered['checkpoint']
            decision = recovered['pending'][0]
            current = self.engine.restore(recovered['checkpoint'])
            self.hooks.hit('before.evaluation')
            result = self.engine.evaluate(current, decision['token'], decision['position'])
            self.hooks.hit('after.evaluation')
            expected = current.history + (decision['token'],)
            if result.position != current.position + 1 or result.history != expected:
                raise Invalid('evaluation token-position drift')
            self.engine.verify(result)
            # QIES094 publisher is synthetic-only and independently computes its fixture blob.
            # A native adapter must publish supplied opaque bytes without evaluating again.
            self.hooks.hit('before.checkpoint_publication')
            published = self.store.evaluate_pending()
            self.hooks.hit('after.checkpoint_publication')
            if published['state'] != result.blob or published['history'] != list(result.history):
                raise Invalid('published state mismatch')
            return published
        except BaseException:
            self.invalid = True
            raise

    def replay(self, start_position=0, end_position=None):
        """Reconstruct committed state without selecting, publishing or emitting."""
        self._valid()
        try:
            self.hooks.hit('before.replay')
            recovered = self.store.recover()
            latest = recovered['checkpoint']['position']
            integer(start_position)
            end = latest if end_position is None else integer(end_position)
            if not start_position <= end <= latest:
                raise Invalid('replay range')
            state = self.engine.restore(checkpoint_at(self.store, start_position))
            self.hooks.hit('after.replay_restore')
            replayed = 0
            for decision in decisions(self.store, recovered['journal_end']):
                position = decision['position']
                if position < start_position:
                    continue
                if position >= end:
                    break
                self.hooks.hit('before.replay_step:%d' % position)
                state = self.engine.evaluate(state, decision['token'], position)
                self.engine.verify(state)
                expected = checkpoint_at(self.store, position + 1)
                if state.position != expected['position'] or list(state.history) != expected['history'] or state.blob != expected['state']:
                    raise Invalid('replayed state mismatch')
                self.hooks.hit('after.replay_step:%d' % position)
                replayed += 1
            if state.position != end:
                raise Invalid('incomplete replay')
            self.hooks.hit('after.replay')
            return {'state': state, 'replayed_tokens': replayed, 'authoritative_position': latest,
                    'pending_token': recovered['pending'][0]['token'] if recovered['pending'] else None,
                    'committed_position_changed': False,
                    'output': {'mode': 'NO_AUTOMATIC_REPLAY', 'delivery': 'UNKNOWN'}}
        except BaseException:
            self.invalid = True
            raise
