#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <string>

// Via std::vector — includes one heap alloc per call (output vector)
extern std::vector<double> ApplyCatboostModelMulti(const std::vector<float>&);
extern double ApplyCatboostModel(const std::vector<float>&);

// Pure computation — no heap allocation, measures inference only
extern double ApplyCatboostModelDirect(float x);

static constexpr int kPregenN = 1 << 14;  // 16k values, power-of-2 for cheap modulo

static std::vector<float> g_inputs;

static void GenerateInputs() {
    if (!g_inputs.empty()) return;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    g_inputs.resize(kPregenN);
    for (auto& v : g_inputs) v = dist(rng);
}

// ── Via vector (includes output allocation) ───────────────────────────────────

static void BM_ViaVector_Single(benchmark::State& state) {
    GenerateInputs();
    std::vector<float> sample(1);
    int idx = 0;
    for (auto _ : state) {
        sample[0] = g_inputs[idx++ & (kPregenN - 1)];
        benchmark::DoNotOptimize(ApplyCatboostModel(sample));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ViaVector_Single);

static void BM_ViaVector_Batch(benchmark::State& state) {
    GenerateInputs();
    const int64_t N = state.range(0);
    std::vector<float> sample(1);
    for (auto _ : state) {
        double acc = 0.0;
        for (int64_t i = 0; i < N; ++i) {
            sample[0] = g_inputs[i & (kPregenN - 1)];
            acc += ApplyCatboostModel(sample);
        }
        benchmark::DoNotOptimize(acc);
    }
    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_ViaVector_Batch)->RangeMultiplier(4)->Range(1, 1 << 14);

// ── Direct (pure inference, no allocation) ────────────────────────────────────

static void BM_Direct_Single(benchmark::State& state) {
    GenerateInputs();
    int idx = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(ApplyCatboostModelDirect(g_inputs[idx++ & (kPregenN - 1)]));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Direct_Single);

static void BM_Direct_Batch(benchmark::State& state) {
    GenerateInputs();
    const int64_t N = state.range(0);
    for (auto _ : state) {
        double acc = 0.0;
        for (int64_t i = 0; i < N; ++i)
            acc += ApplyCatboostModelDirect(g_inputs[i & (kPregenN - 1)]);
        benchmark::DoNotOptimize(acc);
    }
    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_Direct_Batch)->RangeMultiplier(4)->Range(1, 1 << 14);

BENCHMARK_MAIN();
