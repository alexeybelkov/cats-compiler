#include <benchmark/benchmark.h>
#include <random>
#include <vector>

static constexpr int kNumFeatures = 100;
static constexpr int kPregenN = 1 << 10;  // 1024 rows

extern std::vector<double> ApplyCatboostModelMulti(const std::vector<float>&);
extern double ApplyCatboostModel(const std::vector<float>&);
extern double ApplyCatboostModelDirect(const float* features);

// kPregenN rows of kNumFeatures floats, laid out row-major
static std::vector<float> g_inputs;

static void GenerateInputs() {
    if (!g_inputs.empty()) return;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    g_inputs.resize(kPregenN * kNumFeatures);
    for (auto& v : g_inputs) v = dist(rng);
}

// ── Via vector API (includes output allocation) ───────────────────────────────

static void BM_ViaVector_Single(benchmark::State& state) {
    GenerateInputs();
    std::vector<float> sample(kNumFeatures);
    int idx = 0;
    for (auto _ : state) {
        const float* row = g_inputs.data() + ((idx++ & (kPregenN - 1)) * kNumFeatures);
        std::copy(row, row + kNumFeatures, sample.begin());
        benchmark::DoNotOptimize(ApplyCatboostModel(sample));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ViaVector_Single);

static void BM_ViaVector_Batch(benchmark::State& state) {
    GenerateInputs();
    const int64_t N = state.range(0);
    std::vector<float> sample(kNumFeatures);
    for (auto _ : state) {
        double acc = 0.0;
        for (int64_t i = 0; i < N; ++i) {
            const float* row = g_inputs.data() + ((i & (kPregenN - 1)) * kNumFeatures);
            std::copy(row, row + kNumFeatures, sample.begin());
            acc += ApplyCatboostModel(sample);
        }
        benchmark::DoNotOptimize(acc);
    }
    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_ViaVector_Batch)->RangeMultiplier(4)->Range(1, 1 << 10);

// ── Direct (pure inference, no allocation) ────────────────────────────────────

static void BM_Direct_Single(benchmark::State& state) {
    GenerateInputs();
    int idx = 0;
    for (auto _ : state) {
        const float* row = g_inputs.data() + ((idx++ & (kPregenN - 1)) * kNumFeatures);
        benchmark::DoNotOptimize(ApplyCatboostModelDirect(row));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Direct_Single);

static void BM_Direct_Batch(benchmark::State& state) {
    GenerateInputs();
    const int64_t N = state.range(0);
    for (auto _ : state) {
        double acc = 0.0;
        for (int64_t i = 0; i < N; ++i) {
            const float* row = g_inputs.data() + ((i & (kPregenN - 1)) * kNumFeatures);
            acc += ApplyCatboostModelDirect(row);
        }
        benchmark::DoNotOptimize(acc);
    }
    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_Direct_Batch)->RangeMultiplier(4)->Range(1, 1 << 10);

BENCHMARK_MAIN();
