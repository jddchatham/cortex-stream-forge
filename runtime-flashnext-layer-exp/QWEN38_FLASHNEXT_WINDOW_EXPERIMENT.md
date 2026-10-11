# Flash-Next Tensor Window Experiment

This tree is derived from the frozen `FLASHNEXT_SPARSE_MOE_GOLDEN` tree. The
frozen source was not modified.

## Tensor Window result

`flashnext-window-expert136.cpp` tiles complete quantized output rows. It
allocates one aligned arena, uses `pread()` for complete row ranges, binds
each range as a normal 2-D quantized GGML tensor, executes `ggml_mul_mat`,
copies the tile output, synchronizes by completing the backend graph, and
reuses the same arena.

For expert 136, all gate/up/down projections match the non-window executor:

```text
arena       hash
16 KiB      5d5dcaad1155dbf1
64 KiB      5d5dcaad1155dbf1
256 KiB     5d5dcaad1155dbf1
1 MiB       5d5dcaad1155dbf1
4 MiB       5d5dcaad1155dbf1
8 MiB       5d5dcaad1155dbf1
```

At 16 KiB the legal tile rows were 12 for gate/up and 45 for down. The
complete expert read traffic remained 2,662,400 bytes: Tensor Window bounds
residency but does not reduce required selected-expert I/O.

All ten frozen selected experts also pass at 16 KiB. The accumulated selected
hash is `822f120d0c7f2369`, matching the sparse-MoE golden. Combining the
validated shared contribution gives the frozen complete-MoE hash
`c3485d69e3535a65`.

```text
FLASHNEXT EXPERT TENSOR WINDOW: PASS
minimum tested window: 16 KiB
unselected expert reads: 0
```

Complete layer-0 recurrent/state integration has not yet been implemented in
this tree.
