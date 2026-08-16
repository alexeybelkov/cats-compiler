# cats-compiler — Findings & Design Notes

A study project: an LLVM frontend that compiles CatBoost models into optimized
native inference code, using split statistics as branch probabilities.
This document records what we measured and what we concluded about *where*
compiler effort actually pays off.

---

## 1. The models under test

| Model | Features | Trees / depth | Layout | Notes |
|---|---|---|---|---|
| `cb_classifier` (small) | 1 float | 10 trees | flat float | toy; multiclass dim=10 |
| `cb_big_cls` (float-only) | 100 float | 200 × depth-6 | flat float | `BinaryFeatureCount=1112`, `LeafValues[12800][1]` ≈ 100 KB |
| `cb_big_cls` (CTR variant) | 100 float + 100 cat | 200 × depth-6 | hash-map CTR | categorical → runtime `unordered_map` lookups |

CatBoost trees are **oblivious (symmetric)**: every node at a given depth splits
on the *same* feature, so traversal is branchless — binarize features once, then
`index |= binaryFeatures[split] << depth`, then one indexed load from the leaf table.

The final, clean target for the transpiler is the **float-only `cb_big_cls`**:
`CatFeatureCount=0`, no `unordered_map`/`CalcCtrs`, just `Borders[1112]` +
`LeafValues[12800][1]`.

---

## 2. Benchmark setup

Google Benchmark (prebuilt in the LLVM third-party tree). Each model is compiled
two ways and exposed through a shim that defines `ApplyCatboostModelDirect`:

- **classifier shim** — `const auto& model` → loops remain (no unroll)
- **constexpr shim** — `constexpr auto& model` → compiler fully unrolls + constant-propagates

Two measurement modes:
- **ViaVector** — calls the stock `ApplyCatboostModel(std::vector<float>)` API; includes heap allocations.
- **Direct** — calls `ApplyCatboostModelDirect`; pure inference, no allocation.

Build: `-O3 -march=native -DNDEBUG` (clang-19). `-ffast-math` made no measurable difference.

CMake targets: `bench_classifier`, `bench_constexpr`, `bench_big_classifier`,
`bench_big_constexpr`, `bench_ctr`.

---

## 3. Results

### Small model (1 feature, 10 trees) — constexpr is a big win
- Direct: **7.6 ns (constexpr) vs 20.3 ns (loop)** → ~2.66× pure compute.
- IR effects of constexpr: `constant` instead of `zeroinitializer`, full unroll,
  GVN-CSE of the 38 border comparisons, binaryFeatures array eliminated, 1 malloc.

### Big float-only model (100 features, 200 trees) — constexpr does NOT help
| | classifier (loop) | constexpr |
|---|---|---|
| Direct Single | **553 ns** | **614 ns** (slightly *worse*) |
| Direct Batch/1024 | 1.83 M/s | 1.63 M/s |
| ViaVector Single | 711 ns | 589 ns |

- **constexpr stops helping and can hurt**: full unroll of 200 trees / 1112 borders
  blows up the instruction stream → I-cache pressure. The bottleneck is random reads
  into the ~100 KB `LeafValues` table (L2/L3 latency), which unrolling can't touch.
- **constexpr still helps ViaVector** (589 vs 711 ns) — but only because it replaces
  the two `std::vector` allocations (`binaryFeatures`, `results`) with stack arrays.
  That's allocation overhead, not compute.
- **Floor ≈ 550 ns/sample**, set by the leaf-table memory access pattern.

### CTR model (categorical features) — hash maps dominate
| | Single | Throughput |
|---|---|---|
| CTR model | **10,273 ns** | 97 k/s |

- ~18.5× slower than the float-only model.
- ~92% of inference time is the CTR step: ~100 `unordered_map` lookups × ~95 ns each.
- Throughput is flat from batch/1 to batch/1024 → tables never warm in cache.

---

## 4. Key conclusions

1. **For the float-only model, `clang -O3` is already near-optimal.** The bottleneck
   is data access (leaf table), not codegen. constexpr / unrolling / `-ffast-math`
   don't break the ~550 ns floor.

2. **The benchmark "allocation overhead" is real but separate from compute.** The
   stock vector API costs ~136 ns for 2 heap allocations on the big model; stack
   arrays remove it. This is a code-generation choice, not an optimization-pass win.

3. **The CTR model is where a transpiler tells a genuinely different story.** The CTR
   tables are *static, frozen at training time*. clang can't know that — it sees
   `unordered_map`. A transpiler can replace each lookup with a constant array indexed
   by the (bounded) integer category:
   - `unordered_map::find` (~95 ns, cache miss) → `load @ctr_table[val]` (~2 ns, L1).
   - Estimated total: ~10,273 ns → ~950 ns, **~10× overall, ~47× on the CTR step**.
   - **Caveat:** for bounded-integer categoricals this is equally doable in C++ codegen
     (emit `constexpr float table[65]`). LLVM only pulls ahead for *high-cardinality
     string* categoricals (perfect-hash + dense array) or batch SIMD.

---

## 5. Design discussions

### C++ codegen vs LLVM frontend
C++ codegen (intrinsics, `__builtin_expect_with_probability`, `__builtin_prefetch`,
`#pragma unroll`) can express ~95% of what LLVM can do *for this problem*. LLVM wins on:
- **Portability** — one IR → x86 AVX-512 / ARM SVE / RISC-V V, vs per-target intrinsics.
- **Pass composition** — chain dead-leaf-elim → GVN → vectorize → prefetch in order.
- **Analysis-driven decisions** — count zero leaves, decide *whether* a transform pays.
- **Cross-statement value reuse** — GVN *will* CSE duplicate splits; clang *might*.
- **Verifier-checked transforms** — safer than fragile source rewrites.
- **Study value** — the actual point here: learning IR, dominance, GVN, scheduling, passes.

Honest expected gap on the big float model: **10–30%**, mostly from prefetch placement
and guaranteed vectorization. Not dramatic — the *process* is the payoff.

### SIMD
- Single-sample: limited by serial traversal.
- **Batch (8 samples at once)**: vectorize binarization (`fcmp` on 8 lanes) + gather
  leaves. Theoretical ~20–25× vs the CTR model, ~150–200 ns/sample on the float model.
- Why batch SIMD is "hard in C++": a *generator* must commit to a SIMD width
  (`_mm256`/`_mm512`/NEON) and emit `#ifdef` paths per target. In LLVM you emit abstract
  `<8 x float>` once and the backend picks instructions from the target triple.

### Branch optimizations: CatBoost vs XGBoost
- **CatBoost (symmetric)** → already branchless; branch-weight / layout passes are ~irrelevant.
- **XGBoost (asymmetric)** → traversal is a dependent branch chain. Here LLVM branch work
  matters: branchless `select` conversion, profile-guided node reordering, hot-path
  `!prof` block placement, full-tree unrolling. Estimated **20–40%** on traversal.
- So XGBoost is the better target if the goal is *branching* optimizations.

### Profile data
CatBoost JSON has `leaf_weights` per tree (training samples per leaf). For a symmetric
tree, summing leaf weights by bit position at each depth gives exact per-split
probabilities → embeddable as `!prof branch_weights` *without* an instrumented run.
Useful mainly if a model emits real branches (XGBoost), less so for branchless CatBoost.

### MLIR vs LLVM
- LLVM IR = one fixed, near-assembly level.
- MLIR = framework for *building* IRs via **dialects** that coexist and lower
  progressively (e.g. `catboost.tree` → `affine`/`scf` → `llvm`). Has nested regions
  and structured control flow (`scf.for`) that make high-level loop transforms easier.
- For JSON → IR → native on one target: **LLVM is the right level.** MLIR pays off only
  if targeting multiple backends or expressing batch inference as a high-level abstraction.

### Training-time compilation (considered, deprioritized)
The training bottleneck is **histogram building** (scatter-add of gradients into bins),
not scoring (CatBoost updates scores incrementally, O(N × 1 tree) per round). Speeding
that up needs hardware-specific tricks (AVX-512 `vpconflictd`, bin-sorting) — a research
project, not a clean codegen target. **Inference compilation is the better-scoped goal.**

### CTR hash map → tree/array (the user's insight)
A CTR `unordered_map` is a static, read-only mapping at inference time. For bounded
integer categories it collapses to a dense `[N x float]` constant array (one GEP + load,
L1-resident). This is the single transformation clang fundamentally cannot do on the C++
export, because it can't prove the map is frozen — but the transpiler knows from the JSON.

---

## 6. Direction

- **Primary target:** float-only `cb_big_cls`, JSON → LLVM IR transpiler.
  - Python: parse JSON (`oblivious_trees`, `leaf_values`, `leaf_weights`, borders).
  - C++ via pybind11: build & optimize IR with the LLVM API.
- **Realistic wins to chase:** batch SIMD binarization, prefetch into the leaf table.
- **Honest expectation vs `clang -O3`:** 10–30% on this model; the leaf-table memory
  floor (~550 ns) dominates either way.
- **If branching optimizations are the goal:** switch target to XGBoost.
- **If categorical speed is the goal:** the CTR → constant-array transform is the
  highest-leverage single change (~10× on the CTR model).
