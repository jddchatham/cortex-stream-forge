#!/usr/bin/env python3
"""Behavioral mock for the owned-layer coordinator contract."""

class MockCoordinator:
    def __init__(self, start, end, fail=None):
        self.start, self.end, self.fail = start, end, fail
        self.events, self.outputs = [], []

    def run(self):
        previous = [0x3F800000] * 4
        for layer in range(self.start, self.end):
            self.events.append(("load", layer))
            if layer > self.start:
                assert self.events[-2] == ("teardown", layer - 1)
            self.events.append(("input", layer, tuple(previous)))
            if layer == self.fail:
                self.events.append(("fail", layer))
                self.events.append(("teardown", layer))
                return False
            output = [x + layer for x in previous]
            self.outputs.append(tuple(output))
            self.events.append(("commit", layer))
            self.events.append(("teardown", layer))
            previous = output
        return True

def check(start, end):
    c = MockCoordinator(start, end)
    assert c.run()
    assert [x[1] for x in c.events if x[0] == "commit"] == list(range(start, end))
    assert len(c.outputs) == end - start
    return c

def main():
    for end in (2, 3, 4):
        check(1, end)
    failed = MockCoordinator(1, 4, fail=2)
    assert not failed.run()
    assert [x[1] for x in failed.events if x[0] == "commit"] == [1]
    assert failed.events[-1] == ("teardown", 2)
    assert not any(x[0] == "commit" and x[1] == 3 for x in failed.events)
    print("range_coordinator_mock=PASS")

if __name__ == "__main__":
    main()
