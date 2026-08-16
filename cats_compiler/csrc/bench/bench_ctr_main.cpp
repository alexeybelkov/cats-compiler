#include <benchmark/benchmark.h>
#include <random>
#include <string>
#include <vector>

static constexpr int kNumFloat = 100;
static constexpr int kNumCat   = 100;
static constexpr int kCatRange = 64;   // values 1..64
static constexpr int kPregenN  = 1 << 10;

extern double ApplyCatboostModel(
    const std::vector<float>&,
    const std::vector<std::string>&);

// Static string pool "1".."64" — all SSO, no heap allocation on copy
static const std::vector<std::string> kCatPool = []() {
    std::vector<std::string> v;
    v.reserve(kCatRange);
    for (int i = 1; i <= kCatRange; ++i) v.push_back(std::to_string(i));
    return v;
}();

static std::vector<float>   g_float_inputs;  // [kPregenN * kNumFloat]
static std::vector<uint8_t> g_cat_inputs;    // [kPregenN * kNumCat], values 0..63

static void GenerateInputs() {
    if (!g_float_inputs.empty()) return;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> fdist(-3.0f, 3.0f);
    std::uniform_int_distribution<int>    cdist(0, kCatRange - 1);

    g_float_inputs.resize(kPregenN * kNumFloat);
    for (auto& v : g_float_inputs) v = fdist(rng);

    g_cat_inputs.resize(kPregenN * kNumCat);
    for (auto& v : g_cat_inputs) v = static_cast<uint8_t>(cdist(rng));
}

// ── Single sample ─────────────────────────────────────────────────────────────

static void BM_Single(benchmark::State& state) {
    GenerateInputs();
    std::vector<float>       float_sample(kNumFloat);
    std::vector<std::string> cat_sample(kNumCat);
    int idx = 0;
    for (auto _ : state) {
        int row = (idx++ & (kPregenN - 1));
        const float*   frow = g_float_inputs.data() + row * kNumFloat;
        const uint8_t* crow = g_cat_inputs.data()   + row * kNumCat;

        std::copy(frow, frow + kNumFloat, float_sample.begin());
        for (int j = 0; j < kNumCat; ++j)
            cat_sample[j] = kCatPool[crow[j]];

        benchmark::DoNotOptimize(ApplyCatboostModel(float_sample, cat_sample));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Single);

// ── Batch ─────────────────────────────────────────────────────────────────────

static void BM_Batch(benchmark::State& state) {
    GenerateInputs();
    const int64_t N = state.range(0);
    std::vector<float>       float_sample(kNumFloat);
    std::vector<std::string> cat_sample(kNumCat);
    for (auto _ : state) {
        double acc = 0.0;
        for (int64_t i = 0; i < N; ++i) {
            int row = (i & (kPregenN - 1));
            const float*   frow = g_float_inputs.data() + row * kNumFloat;
            const uint8_t* crow = g_cat_inputs.data()   + row * kNumCat;

            std::copy(frow, frow + kNumFloat, float_sample.begin());
            for (int j = 0; j < kNumCat; ++j)
                cat_sample[j] = kCatPool[crow[j]];

            acc += ApplyCatboostModel(float_sample, cat_sample);
        }
        benchmark::DoNotOptimize(acc);
    }
    state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_Batch)->RangeMultiplier(4)->Range(1, 1 << 10);

BENCHMARK_MAIN();
