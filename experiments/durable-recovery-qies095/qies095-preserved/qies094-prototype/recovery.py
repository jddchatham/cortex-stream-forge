"""Model-free durable state protocol. No QUORTEX/model imports or stdout delivery."""
import base64
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import time

MAGIC = b'QIES094J'
HEADER = struct.Struct('<8sHI')
VERSION = 1
MAX_FRAME = 2 * 1024 * 1024
MAX_STEPS = 64
MAX_FILE = 32768
IDENTITY = 'SYNTHETIC-STATE-v1'
ZERO = '0' * 64


class Invalid(Exception):
    pass


class Crash(BaseException):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise Invalid('duplicate JSON key')
            result[key] = value
        return result
    try:
        result = json.loads(data, object_pairs_hook=pairs)
        if encode(result) != data:
            raise Invalid('noncanonical JSON')
        return result
    except (ValueError, TypeError, RecursionError) as exc:
        raise Invalid('invalid JSON') from exc


def integer(value, limit=MAX_STEPS):
    if type(value) is not int or not 0 <= value <= limit:
        raise Invalid('integer bounds')
    return value


def synthetic(history):
    return ('attention|recurrent|indexer|' + digest(encode(history))) * 8


class Hooks:
    def __init__(self, at=None, action=None, record=False):
        self.events = []
        self.count = 0
        self.record = record
        self.at = at
        self.action = action

    def hit(self, label):
        self.count += 1
        if self.record:
            self.events.append(label)
        if self.count == self.at:
            if self.action:
                self.action()
            raise Crash(label)


class MemoryFS:
    """Independent file-data and directory-name persistence, one flat directory."""
    def __init__(self, hooks=None):
        self.hooks = hooks or Hooks()
        self.names = {}
        self.data = {}
        self.durable_names = {}
        self.durable_data = {}
        self.serial = 0

    def write(self, name, data, append=False):
        self.hooks.hit('before.write:' + name)
        if append:
            if name not in self.names:
                raise Invalid('missing append file')
        else:
            if name in self.names:
                raise Invalid('exclusive creation')
            self.serial += 1
            self.names[name] = self.serial
            self.data[self.serial] = b''
        inode = self.names[name]
        half = max(1, len(data) // 2)
        self.data[inode] += data[:half]
        self.hooks.hit('mid.write:' + name)
        self.data[inode] += data[half:]
        self.hooks.hit('after.write:' + name)
        self.hooks.hit('after.close:' + name)

    def sync(self, name):
        self.hooks.hit('before.fsync:' + name)
        inode = self.names[name]
        self.durable_data[inode] = self.data[inode]
        self.hooks.hit('after.fsync:' + name)

    def syncdir(self):
        self.hooks.hit('before.dirsync')
        self.durable_names = self.names.copy()
        self.hooks.hit('after.dirsync')

    def rename(self, old, new):
        self.hooks.hit('before.rename:' + new)
        if new in self.names:
            raise Invalid('rename collision')
        self.names[new] = self.names.pop(old)
        self.hooks.hit('after.rename:' + new)

    def read(self, name, limit=MAX_FILE):
        self.hooks.hit('scan.read:' + name)
        if name not in self.names:
            raise Invalid('missing required file: ' + name)
        data = self.data[self.names[name]]
        if len(data) > limit:
            raise Invalid('file capacity')
        return data

    def names_list(self):
        self.hooks.hit('scan.names')
        if len(self.names) > 512:
            raise Invalid('directory capacity')
        return sorted(self.names)

    def reboot(self, retain=False, names_survive=None, data_survive=None):
        fs = MemoryFS()
        # Both total loss of unsynced changes and total survival are tested.
        names_survive = retain if names_survive is None else names_survive
        data_survive = retain if data_survive is None else data_survive
        fs.names = self.names.copy() if names_survive else self.durable_names.copy()
        fs.data = self.data.copy() if data_survive else {
            inode: self.durable_data.get(inode, b'') for inode in fs.names.values()}
        fs.serial = self.serial
        fs.durable_names = fs.names.copy()
        fs.durable_data = fs.data.copy()
        return fs


class RealFS:
    def __init__(self, root, hooks=None):
        self.root = Path(root)
        self.hooks = hooks or Hooks()
        if not self.root.is_dir() or self.root.is_symlink():
            raise Invalid('dedicated existing directory required')
        self.dirfd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.lockfd = os.open('writer.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=self.dirfd)
        try:
            fcntl.flock(self.lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self.lockfd)
            os.close(self.dirfd)
            raise
        # Persist root creation in its parent as well as entries in root.
        parent = os.open(self.root.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)

    def close(self):
        os.close(self.lockfd)
        os.close(self.dirfd)

    def write(self, name, data, append=False):
        self.hooks.hit('before.write:' + name)
        flags = os.O_WRONLY | os.O_NOFOLLOW | (os.O_APPEND if append else os.O_CREAT | os.O_EXCL)
        fd = os.open(name, flags, 0o600, dir_fd=self.dirfd)
        try:
            half = max(1, len(data) // 2)
            for index, part in enumerate((data[:half], data[half:])):
                view = memoryview(part)
                while view:
                    try:
                        size = os.write(fd, view)
                    except InterruptedError:
                        continue
                    if size <= 0:
                        raise OSError('zero write')
                    view = view[size:]
                if index == 0:
                    self.hooks.hit('mid.write:' + name)
            self.hooks.hit('after.write:' + name)
        finally:
            os.close(fd)
        self.hooks.hit('after.close:' + name)

    def sync(self, name):
        self.hooks.hit('before.fsync:' + name)
        fd = os.open(name, os.O_RDWR | os.O_NOFOLLOW, dir_fd=self.dirfd)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.hooks.hit('after.fsync:' + name)

    def syncdir(self):
        self.hooks.hit('before.dirsync')
        os.fsync(self.dirfd)
        self.hooks.hit('after.dirsync')

    def rename(self, old, new):
        self.hooks.hit('before.rename:' + new)
        # Exclusive writer and private directory; no external concurrent mutation supported.
        try:
            os.stat(new, dir_fd=self.dirfd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise Invalid('rename collision')
        os.rename(old, new, src_dir_fd=self.dirfd, dst_dir_fd=self.dirfd)
        self.hooks.hit('after.rename:' + new)

    def read(self, name, limit=MAX_FILE):
        self.hooks.hit('scan.read:' + name)
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.dirfd)
        try:
            if os.fstat(fd).st_size > limit:
                raise Invalid('file capacity')
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                data = stream.read(limit + 1)
            if len(data) > limit:
                raise Invalid('file capacity')
            return data
        finally:
            os.close(fd)

    def names_list(self):
        self.hooks.hit('scan.names')
        result = []
        scanfd = os.open('.', os.O_RDONLY | os.O_DIRECTORY, dir_fd=self.dirfd)
        try:
            with os.scandir(scanfd) as entries:
                for entry in entries:
                    result.append(entry.name)
                    if len(result) > 512:
                        raise Invalid('directory capacity')
        finally:
            os.close(scanfd)
        return sorted(result)


class Store:
    def __init__(self, fs, session):
        if not isinstance(session, str) or not session.isalnum() or len(session) > 32:
            raise Invalid('session identity')
        self.fs = fs
        self.session = session
        self.poisoned = False
        self.last_journal_ns = 0
        self.last_checkpoint_ns = 0

    def initialize(self):
        if any(n != 'writer.lock' for n in self.fs.names_list()):
            raise Invalid('new store required')
        self.fs.write('identity', encode({'session': self.session, 'schema': VERSION, 'identity': IDENTITY}))
        self.fs.sync('identity')
        self.fs.write('journal', b'')
        self.fs.sync('journal')
        self.fs.syncdir()
        self._publish([], None, 0, ZERO, ZERO)

    def _publish(self, history, parent, offset, journal_hash, decision_hash):
        start = time.perf_counter_ns()
        pos = len(history)
        names = self.fs.names_list()
        attempt = 0
        while any(n.startswith('checkpoint-%03d-%03d' % (pos, attempt)) or n == 'commit-%03d-%03d.tmp' % (pos, attempt) for n in names):
            attempt += 1
            if attempt > 128:
                raise Invalid('attempt capacity')
        name = 'checkpoint-%03d-%03d' % (pos, attempt)
        checkpoint = {'generation': name, 'schema': VERSION, 'identity': IDENTITY, 'session': self.session,
                      'position': pos, 'history': history, 'state': synthetic(history),
                      'parent': parent, 'journal_offset': offset, 'journal_hash': journal_hash,
                      'decision_hash': decision_hash}
        blob = encode(checkpoint)
        self.fs.write(name + '.tmp', blob)
        self.fs.sync(name + '.tmp')
        self.fs.rename(name + '.tmp', name)
        self.fs.syncdir()
        if self.fs.read(name) != blob:
            raise Invalid('checkpoint readback')
        receipt = encode({'schema': VERSION, 'session': self.session, 'position': pos,
                          'checkpoint': digest(blob), 'file': name, 'parent': parent})
        target = 'commit-%03d' % pos
        temporary = target + '-%03d.tmp' % attempt
        self.fs.write(temporary, receipt)
        self.fs.sync(temporary)
        self.fs.rename(temporary, target)
        self.fs.syncdir()
        self.fs.hooks.hit('coordinator.committed')
        self.last_checkpoint_ns = time.perf_counter_ns() - start
        return checkpoint

    def recover(self):
        fs = self.fs
        identity = fs.read('identity')
        if identity != encode({'session': self.session, 'schema': VERSION, 'identity': IDENTITY}):
            raise Invalid('identity mismatch')
        names = fs.names_list()
        commits = []
        for name in names:
            if name.startswith('commit-') and not name.endswith('.tmp'):
                if len(name) != 10 or not name[7:].isdigit():
                    raise Invalid('receipt name')
                commits.append(name)
        if not commits or len(commits) > MAX_STEPS + 1:
            raise Invalid('missing genesis or capacity')
        # Journal scan is bounded by the configured 64-token fixture quota.
        raw = fs.read('journal', (MAX_STEPS + 1) * 8192)
        decisions = []
        offset = 0
        previous = ZERO
        prefix = {0: ZERO}
        torn = False
        while offset < len(raw):
            fs.hooks.hit('scan.frame')
            if len(raw) - offset < HEADER.size:
                torn = True
                break
            magic, version, size = HEADER.unpack_from(raw, offset)
            if magic != MAGIC or version != VERSION or size > MAX_FRAME:
                raise Invalid('frame header')
            end = offset + HEADER.size + size + 32
            if end > len(raw):
                torn = True
                break
            frame = raw[offset:end-32]
            checksum = raw[end-32:end].hex()
            if digest(frame) != checksum:
                raise Invalid('journal checksum')
            d = decode(raw[offset+HEADER.size:end-32])
            expected = {'schema', 'session', 'sequence', 'position', 'token', 'checkpoint', 'previous', 'piece'}
            if set(d) != expected or type(d['schema']) is not int or d['schema'] != VERSION or d['session'] != self.session or d['previous'] != previous:
                raise Invalid('decision identity')
            seq = integer(d['sequence'])
            if seq != len(decisions) + 1 or integer(d['position']) != seq - 1:
                raise Invalid('decision sequence')
            integer(d['token'], 255)
            try:
                piece = base64.b64decode(d['piece'], validate=True)
            except (ValueError, TypeError) as exc:
                raise Invalid('piece encoding') from exc
            if len(piece) > 1024 or d['piece'] != base64.b64encode(piece).decode():
                raise Invalid('fixture piece quota')
            decisions.append((d, end, checksum))
            offset = end
            previous = checksum
            prefix[end] = checksum
        previous_receipt = None
        checkpoint_hashes = []
        current = None
        for pos, name in enumerate(commits):
            if pos > MAX_STEPS or name != 'commit-%03d' % pos:
                raise Invalid('receipt sequence')
            receipt_bytes = fs.read(name)
            receipt = decode(receipt_bytes)
            file = receipt.get('file', '')
            if not isinstance(file, str) or len(file) != 18 or not file.startswith('checkpoint-%03d-' % pos) or not file[-3:].isdigit():
                raise Invalid('checkpoint filename')
            cp_bytes = fs.read(file)
            cp = decode(cp_bytes)
            if receipt_bytes != encode( {'schema': VERSION, 'session': self.session, 'position': pos,
                           'checkpoint': digest(cp_bytes), 'file': file, 'parent': previous_receipt}):
                raise Invalid('receipt mismatch')
            history = [d[0]['token'] for d in decisions[:pos]]
            boundary = decisions[pos-1][1] if pos else 0
            chain = decisions[pos-1][2] if pos else ZERO
            expected_cp = {'generation': file, 'schema': VERSION, 'identity': IDENTITY, 'session': self.session,
                           'position': pos, 'history': history, 'state': synthetic(history),
                           'parent': previous_receipt, 'journal_offset': boundary,
                           'journal_hash': chain, 'decision_hash': chain}
            if len(history) != pos or cp_bytes != encode(expected_cp) or prefix.get(boundary) != chain:
                raise Invalid('checkpoint state/prefix')
            if pos and decisions[pos-1][0]['checkpoint'] != checkpoint_hashes[pos-1]:
                raise Invalid('decision checkpoint reference')
            checkpoint_hashes.append(digest(cp_bytes))
            previous_receipt = digest(receipt_bytes)
            current = cp
        pos = current['position']
        if len(decisions) not in (pos, pos + 1):
            raise Invalid('multiple pending decisions')
        pending = decisions[pos] if len(decisions) > pos else None
        if pending and pending[0]['checkpoint'] != checkpoint_hashes[pos]:
            raise Invalid('pending checkpoint reference')
        return {'checkpoint': current, 'checkpoint_hash': checkpoint_hashes[-1],
                'receipt_hash': previous_receipt, 'pending': pending, 'torn_tail': torn,
                'journal_end': offset, 'journal_hash': previous,
                'output_delivery': 'UNKNOWN; no external acknowledgment; no automatic replay'}

    def prepare(self, token, piece):
        if self.poisoned:
            raise Invalid('failed writer; reopen and recover')
        try:
            state = self.recover()
            if state['torn_tail'] or state['pending']:
                raise Invalid('tail or pending decision; no append permitted')
            pos = state['checkpoint']['position']
            if pos >= MAX_STEPS or len(piece) > 1024:
                raise Invalid('capacity')
            integer(token, 255)
            decision = {'schema': VERSION, 'session': self.session, 'sequence': pos + 1,
                        'position': pos, 'token': token, 'checkpoint': state['checkpoint_hash'],
                        'previous': state['journal_hash'], 'piece': base64.b64encode(piece).decode()}
            payload = encode(decision)
            frame = HEADER.pack(MAGIC, VERSION, len(payload)) + payload
            frame += hashlib.sha256(frame).digest()
            start = time.perf_counter_ns()
            self.fs.hooks.hit('before.append')
            self.fs.write('journal', frame, append=True)
            self.fs.sync('journal')
            self.fs.hooks.hit('decision.durable')
            self.last_journal_ns = time.perf_counter_ns() - start
            return decision
        except BaseException:
            self.poisoned = True
            raise

    def evaluate_pending(self):
        if self.poisoned:
            raise Invalid('failed writer; reopen and recover')
        try:
            state = self.recover()
            if state['torn_tail'] or not state['pending']:
                raise Invalid('pending decision required, intact journal required')
            d, end, checksum = state['pending']
            history = state['checkpoint']['history'] + [d['token']]
            return self._publish(history, state['receipt_hash'], end, checksum, checksum)
        except BaseException:
            self.poisoned = True
            raise
