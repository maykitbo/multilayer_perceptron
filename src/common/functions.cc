#include "strategy.h"

using namespace s21;

fp_type Func::XavierWeightsInit(int rows, int cols) {
    fp_type variance = std::sqrt(xavier / (fp_type)(rows + cols));
    return s21::Random::Normal<fp_type>(0.0, variance);
}

fp_type Func::NormalWeightsInit(fp_type mean, fp_type sd) {
    return s21::Random::Normal<fp_type>(mean, sd);
}

fp_type Func::ActivationReLU(const fp_type x) {
    return x > 0 ? x : 0;
}

fp_type Func::DerivativeActivationReLU(const fp_type x) {
    return x > 0 ? 1 : 0;
}

fp_type Func::ActivationSigmoid(const fp_type x) {
    return 1.0 / (1.0 + std::exp(-1.0 * x));
}

fp_type Func::DerivativeActivationSigmoid(const fp_type x) {
    return std::exp(-1.0 * x) / std::pow(std::exp(-1.0 * x) + 1.0, 2.0);
}

fp_type Func::ActivationSiLU(const fp_type x) {
    return x / (1.0 + std::exp(-1.0 * x));
}

fp_type Func::DerivativeActivationSiLU(const fp_type x) {
    return (1.0 + std::exp(-1.0 * x) + x * std::exp(-1.0 * x)) / std::pow(1.0 + std::exp(-1.0 * x), 2.0);
}

fp_type Func::MeanError(const std::vector<fp_type> &error) {
    return std::sqrt(std::accumulate(error.begin(), error.end(), 0.0, [](fp_type a, fp_type b) {
        return a + std::pow(b, 2);
    }) / (fp_type)error.size());
}
