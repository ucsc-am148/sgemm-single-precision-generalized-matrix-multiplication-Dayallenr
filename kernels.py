"""Student kernels for the SGEMM autograder assignment.

You implement K2 (GMEM coalescing), K3 (shared-memory blocking), K4 (1D
register tiling), and K5 (2D register tiling) inside this file. The launch
wrappers, tile-size constants, and signatures are provided — you only edit
the kernel bodies marked TODO.

K1 (naive) is given as a worked example so you have a reference for the
numba.cuda @cuda.jit signature every kernel must match.

To check correctness locally before submitting:
    python sanity_check.py

To submit: push your edits to the main branch of this assignment repo.
Each push that touches kernels.py triggers the autograder, which runs
on a Modal A100 40GB and posts your grade as a comment on the commit.
You have 5 graded submissions per assignment.
"""
import math

from numba import cuda, float32


# ── Tile constants ──────────────────────────────────────────────────
# These are tied to the launch shapes the autograder will use. Do not
# change them; the run_kN wrappers below depend on these values.

BLOCKSIZE = 32          # K1 + K2 tile

# K3 tile sizes
BM3, BN3, BK3 = 32, 32, 32

# K4 tile sizes
BM4, BN4, BK4 = 64, 64, 8
TM4 = 8

# K5 tile sizes
BM5, BN5, BK5 = 128, 128, 8
TM5, TN5 = 8, 8


# ── K1: naive (worked example, do not edit) ─────────────────────────

@cuda.jit
def sgemm_naive(A, B, C, M, N, K):
    """K1: one thread per output element. No tiling, no shared memory.
    Provided so you have a working numba.cuda kernel for reference.
    """
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x < M and y < N:
        tmp = float32(0.0)
        for i in range(K):
            tmp += A[x, i] * B[i, y]
        C[x, y] = tmp


# ── K2: GMEM coalescing (TODO) ──────────────────────────────────────

@cuda.jit
def sgemm_coalesced(A, B, C, M, N, K):
    """K2: rewrite K1 so that 32 threads in a warp end up writing to 32
    *consecutive columns* of C (and reading 32 consecutive elements of B).
    The arithmetic is identical to K1

    Launch shape (run_k2 below uses this):
        block = (BLOCKSIZE * BLOCKSIZE,)        # 1024 threads, 1D
        grid  = (ceil(M / BLOCKSIZE), ceil(N / BLOCKSIZE))

    With a 1D block of 1024 threads, threadIdx.x runs 0..1023.
    Derive (row_in_tile, col_in_tile) from threadIdx.x using integer division
    and modulo by BLOCKSIZE. 
    Be careful which one indexes the column.
    """
    # col = tid % BLOCKSIZE so consecutive threads hit consecutive columns to coalesced
    row_in_tile = cuda.threadIdx.x // BLOCKSIZE
    col_in_tile = cuda.threadIdx.x % BLOCKSIZE

    x = cuda.blockIdx.x * BLOCKSIZE + row_in_tile
    y = cuda.blockIdx.y * BLOCKSIZE + col_in_tile

    if x < M and y < N:
        tmp = float32(0.0)
        for i in range(K):
            tmp += A[x, i] * B[i, y]
        C[x, y] = tmp


# ── K3: shared-memory cache-blocking (TODO) ─────────────────────────

@cuda.jit
def sgemm_smem(A, B, C, M, N, K):
    """K3: stream the K dimension in chunks of BK3. Each block computes a
            BM3 x BN3 output tile by repeatedly:
        1. cooperatively loading a BM3 x BK3 slice of A and a BK3 x BN3
           slice of B into shared memory (one element per thread per slice),
        2. cuda.syncthreads(),
        3. dotting the row of As into the column of Bs to update one
           per-thread accumulator,
        4. cuda.syncthreads() before the next K-chunk.

    Launch shape (run_k3 below uses this):
        block = (BM3 * BN3,)                    # 1024 threads, 1D
        grid  = (ceil(M / BM3), ceil(N / BN3))

    Use cuda.shared.array((BM3, BK3), float32) for As and a similar
    (BK3, BN3) for Bs.
    Use 0.0 in the SMEM load when the global index is out of bounds.
    """
    As = cuda.shared.array((BM3, BK3), float32)
    Bs = cuda.shared.array((BK3, BN3), float32)

    # Each thread's position within the block tile
    tid = cuda.threadIdx.x

    # row within the BM3 x BN3 output tile
    local_row = tid // BN3

    # col within the BM3 x BN3 output tile
    local_col = tid % BN3

    global_row = cuda.blockIdx.x * BM3 + local_row
    global_col = cuda.blockIdx.y * BN3 + local_col

    acc = float32(0.0)

    # Stream K in chunks of BK3
    num_chunks = (K + BK3 - 1) // BK3
    for chunk in range(num_chunks):
        k_start = chunk * BK3

        # since BM3 * BK3 == BM3 * BN3 == 1024 == block size, one-to-one.
        a_row = cuda.blockIdx.x * BM3 + local_row
        a_col = k_start + local_col
        if a_row < M and a_col < K:
            As[local_row, local_col] = A[a_row, a_col]
        else:
            As[local_row, local_col] = float32(0.0)

        # Reuse local_row as the k-index within the chunk, local_col as the n-index.
        b_row = k_start + local_row
        b_col = cuda.blockIdx.y * BN3 + local_col
        if b_row < K and b_col < N:
            Bs[local_row, local_col] = B[b_row, b_col]
        else:
            Bs[local_row, local_col] = float32(0.0)

        cuda.syncthreads()

        # Accumulate dot product
        for dk in range(BK3):
            acc += As[local_row, dk] * Bs[dk, local_col]

        cuda.syncthreads()

    if global_row < M and global_col < N:
        C[global_row, global_col] = acc


# ── K4: 1D register tiling (TODO) ───────────────────────────────────

@cuda.jit
def sgemm_1d_tile(A, B, C, M, N, K):
    """K4: extend K3 by giving each thread TM4 = 8 rows in a single column
    of the BM4 x BN4 output tile.

    Note: blockIdx.x now indexes COLUMNS of the output.
    The run_k4 wrapper below already accounts for this, but you need to compute the global (row, col)
    start of your block accordingly.

    Launch shape (run_k4 below uses this):
        block = ((BM4 * BN4) // TM4,)           # 512 threads
        grid  = (ceil(N / BN4), ceil(M / BM4))  # x = col, y = row

    Cooperative loads here are tidy: A's tile is BM4 x BK4 = 512 elements,
    B's tile is BK4 x BN4 = 512 elements, and you have 512 threads so
    exactly one element per thread per tile (so no inner-load loop)

    Use cuda.local.array(TM4, float32) for the per-thread accumulator array.
    Initialize all entries to 0.0 before the K-loop.
    """
 
    As = cuda.shared.array((BM4, BK4), float32)
    Bs = cuda.shared.array((BK4, BN4), float32)
 
    tid = cuda.threadIdx.x

    # blockIdx.x is cols, blockIdx.y is rows (axis swap from K3)
    # each thread owns one column and TM4 consecutive rows in the output tile
    thread_col = tid % BN4
    thread_row = (tid // BN4) * TM4
 
    # Block origins in global C
    block_row = cuda.blockIdx.y * BM4
    block_col = cuda.blockIdx.x * BN4
 
    # TM4 accumulators, one per row this thread owns
    acc = cuda.local.array(TM4, float32)
    for i in range(TM4):
        acc[i] = float32(0.0)
 
   # 512 threads, 512 elements per tile, so one element per thread per load
    a_load_row = tid // BK4
    a_load_col = tid % BK4
    b_load_row = tid // BN4
    b_load_col = tid % BN4
 
    num_chunks = (K + BK4 - 1) // BK4
    for chunk in range(num_chunks):
        k_start = chunk * BK4
 
        # Load A and B tiles
        ga_row = block_row + a_load_row
        ga_col = k_start + a_load_col
        if ga_row < M and ga_col < K:
            As[a_load_row, a_load_col] = A[ga_row, ga_col]
        else:
            As[a_load_row, a_load_col] = float32(0.0)
 
        gb_row = k_start + b_load_row
        gb_col = block_col + b_load_col
        if gb_row < K and gb_col < N:
            Bs[b_load_row, b_load_col] = B[gb_row, gb_col]
        else:
            Bs[b_load_row, b_load_col] = float32(0.0)
 
        cuda.syncthreads()
 
        # broadcast one B value per dk step, FMA into all TM4 accumulators
        for dk in range(BK4):
            b_val = Bs[dk, thread_col]
            for tm in range(TM4):
                acc[tm] += As[thread_row + tm, dk] * b_val
 
        cuda.syncthreads()
 
    # Write TM4 results
    for tm in range(TM4):
        grow = block_row + thread_row + tm
        gcol = block_col + thread_col
        if grow < M and gcol < N:
            C[grow, gcol] = acc[tm]
    return


# ── K5: 2D register tiling (TODO) ───────────────────────────────────

@cuda.jit
def sgemm_2d_tile(A, B, C, M, N, K):
    """K5: extend K4 to a TM5 x TN5 = 8 x 8 register tile per thread.
    Inside the inner-k loop, cache TM5 As values and TN5 Bs values into
    register arrays, then do the TM5 x TN5 outer-product update.

    Launch shape (run_k5 below uses this):
        block = ((BM5 * BN5) // (TM5 * TN5),)   # 256 threads
        grid  = (ceil(N / BN5), ceil(M / BM5))

    Cooperative loads now need a stride loop: the tile has more elements
    (BM5 * BK5 = 1024) than the block has threads (256), so each thread
    loads BM5 * BK5 / 256 = 4 elements of A per K-chunk and similarly for B.
    Pick the per-thread row stride so that consecutive threads touch
    consecutive memory addresses (= coalesced GMEM loads).

    For accumulators, use cuda.local.array((TM5, TN5), float32).
    Numba supports tuple-shaped local arrays!
    """
    As = cuda.shared.array((BM5, BK5), float32)
    Bs = cuda.shared.array((BK5, BN5), float32)

    # each thread owns a TM5 x TN5 sub tile of the output
    # 256 threads arranged as 16x16, each covering an 8x8 patch
    tid = cuda.threadIdx.x
    num_thread_cols = BN5 // TN5       # 16
    thread_col = tid % num_thread_cols
    thread_row = tid // num_thread_cols

    block_row = cuda.blockIdx.y * BM5
    block_col = cuda.blockIdx.x * BN5

    # TM5 x TN5 accumulators for this threads output patch
    acc = cuda.local.array((TM5, TN5), float32)
    for tm in range(TM5):
        for tn in range(TN5):
            acc[tm, tn] = float32(0.0)

    reg_a = cuda.local.array(TM5, float32)
    reg_b = cuda.local.array(TN5, float32)

    # tile has 1024 elements but only 256 threads, so each thread loads 4
    # striding in rows keeps consecutive threads on consecutive columns (coalesced)
    num_threads = (BM5 * BN5) // (TM5 * TN5)  # 256
    a_inner_col = tid % BK5
    a_inner_row = tid // BK5
    a_row_stride = num_threads // BK5

    b_inner_col = tid % BN5
    b_inner_row = tid // BN5
    b_row_stride = num_threads // BN5

    # 4 elements per thread
    a_loads = (BM5 * BK5) // num_threads  # 4
    b_loads = (BK5 * BN5) // num_threads  # 4

    num_chunks = (K + BK5 - 1) // BK5
    for chunk in range(num_chunks):
        k_start = chunk * BK5

        # Load A tile
        for i in range(a_loads):
            local_row = a_inner_row + i * a_row_stride
            local_col = a_inner_col
            ga_row = block_row + local_row
            ga_col = k_start + local_col
            if ga_row < M and ga_col < K:
                As[local_row, local_col] = A[ga_row, ga_col]
            else:
                As[local_row, local_col] = float32(0.0)

        # Load B tile
        for i in range(b_loads):
            local_row = b_inner_row + i * b_row_stride
            local_col = b_inner_col
            gb_row = k_start + local_row
            gb_col = block_col + local_col
            if gb_row < K and gb_col < N:
                Bs[local_row, local_col] = B[gb_row, gb_col]
            else:
                Bs[local_row, local_col] = float32(0.0)

        cuda.syncthreads()

        # Inner k loop: outer-product update
        for dk in range(BK5):
            # Cache TM5 A values into registers
            for tm in range(TM5):
                reg_a[tm] = As[thread_row * TM5 + tm, dk]
            # Cache TN5 B values into registers
            for tn in range(TN5):
                reg_b[tn] = Bs[dk, thread_col * TN5 + tn]
            # TM5 x TN5 outer-product update
            for tm in range(TM5):
                for tn in range(TN5):
                    acc[tm, tn] += reg_a[tm] * reg_b[tn]

        cuda.syncthreads()

    # Write results
    for tm in range(TM5):
        for tn in range(TN5):
            grow = block_row + thread_row * TM5 + tm
            gcol = block_col + thread_col * TN5 + tn
            if grow < M and gcol < N:
                C[grow, gcol] = acc[tm, tn]


# ── Launch wrappers (provided — do not edit) ────────────────────────

def run_k1(A, B, C, M, N, K):
    grid = (math.ceil(M / BLOCKSIZE), math.ceil(N / BLOCKSIZE))
    block = (BLOCKSIZE, BLOCKSIZE)
    sgemm_naive[grid, block](A, B, C, M, N, K)


def run_k2(A, B, C, M, N, K):
    grid = (math.ceil(M / BLOCKSIZE), math.ceil(N / BLOCKSIZE))
    block = (BLOCKSIZE * BLOCKSIZE,)
    sgemm_coalesced[grid, block](A, B, C, M, N, K)


def run_k3(A, B, C, M, N, K):
    grid = (math.ceil(M / BM3), math.ceil(N / BN3))
    block = (BM3 * BN3,)
    sgemm_smem[grid, block](A, B, C, M, N, K)


def run_k4(A, B, C, M, N, K):
    # Axis swap: blockIdx.x indexes columns of C.
    grid = (math.ceil(N / BN4), math.ceil(M / BM4))
    block = ((BM4 * BN4) // TM4,)
    sgemm_1d_tile[grid, block](A, B, C, M, N, K)


def run_k5(A, B, C, M, N, K):
    grid = (math.ceil(N / BN5), math.ceil(M / BM5))
    block = ((BM5 * BN5) // (TM5 * TN5),)
    sgemm_2d_tile[grid, block](A, B, C, M, N, K)


# Graded kernels in the order the rubric uses (1/4 → C, 2/4 → B-, ...).
KERNELS = [
    ("k2_coalesce", run_k2),
    ("k3_smem",     run_k3),
    ("k4_1d_tile",  run_k4),
    ("k5_2d_tile",  run_k5),
]