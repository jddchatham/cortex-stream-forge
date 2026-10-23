# Quortex: contextual convergence experiment

Quortex explores a distinction between *completeness* and *correctness*. A candidate can be structurally complete (1/1) but fail a time-sensitive or task-specific constraint.

## Quadrant
Two independently evaluated binary features A and B:
- 0/0: neither present
- 1/0: A only
- 0/1: B only
- 1/1: both present (candidate, not acceptance)

These are feature states, **not arithmetic fractions**. Contextual officiation checks whether a complete candidate meets explicit requirements. If not, it must not converge.

## First implementation assignment for Codex
1. Inspect the existing Cortex Stream Forge architecture and benchmark scripts before changing inference code.
2. Implement a small standalone deterministic prototype (prefer existing repo language and conventions): QuadrantState, ContextRequirements, Candidate, Officiator, ConvergenceController. Return explicit reasons for accept/reject/insufficient-information; do not silently accept incomplete contexts.
3. Add tests: dollar-store basketball is a basketball but fails official-2026-NBA requirement; official-2026 candidate fails official-2027 requirement; ordinary birthday request may accept a non-official basketball; 1/0, 0/1, 0/0 do not count as complete.
4. Record versioned context and explain how changes invalidate previous acceptance.
5. Add a benchmark plan comparing exact model output, accuracy, tokens/sec, RAM, disk I/O, and total latency against unmodified Cortex. Do not claim inference acceleration from the validator alone.
6. Keep the existing streaming inference path and its correctness tests intact. Do not change weight loading or numerical kernels until an experimentally testable computation-saving hypothesis is specified.

## Research question
Can cheap, context-aware candidate rejection or reuse reduce expensive model computation while preserving an agreed reference output? The initial quadrant prototype demonstrates semantics only; it does not prove quantum computation or faster inference.
