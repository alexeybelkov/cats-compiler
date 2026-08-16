#define main _cb_constexpr_main_unused
#include "../../../artifacts/models/cb_constexpr.cpp"
#undef main

// Pure inference — no std::vector, no heap allocation.
// CatboostModelStatic is constexpr, so the compiler fully unrolls both loops,
// constant-propagates all split indices, and GVN-CSEs duplicate comparisons.
double ApplyCatboostModelDirect(float x) {
    constexpr auto& model = CatboostModelStatic;

    unsigned char binaryFeatures[model.BinaryFeatureCount];
    unsigned int binFeatureIndex = 0;
    for (unsigned int i = 0; i < model.FloatFeatureCount; ++i) {
        for (unsigned int j = 0; j < model.BorderCounts[i]; ++j) {
            binaryFeatures[binFeatureIndex] = (unsigned char)(x > model.Borders[binFeatureIndex]);
            ++binFeatureIndex;
        }
    }

    double result = 0.0;
    const unsigned int* treeSplitsPtr = model.TreeSplits;
    const auto* leafValuesPtr = model.LeafValues;
    for (unsigned int treeId = 0; treeId < model.TreeCount; ++treeId) {
        const unsigned int depth = model.TreeDepth[treeId];
        unsigned int index = 0;
        for (unsigned int d = 0; d < depth; ++d)
            index |= (binaryFeatures[treeSplitsPtr[d]] << d);
        result += leafValuesPtr[index][0];
        treeSplitsPtr += depth;
        leafValuesPtr += 1 << depth;
    }
    return model.Scale * result + model.Biases[0];
}
