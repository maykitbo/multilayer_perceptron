#pragma once

#include "types.h"
#include <math.h>
#include "../lib/random.h"
#include "types.h"
#include <fstream>
#include <numeric>

namespace s21 {

/**
 * @enum ActivationFunctions
 * @brief An enumeration representing different activation functions for neural networks.
 */
enum ActivationFunctions {
    kSigmoid, ///< Sigmoid activation function.
    kSiLU,    ///< SiLU (Sigmoid Linear Unit) activation function.
    kReLU,    ///< Rectified Linear Unit (ReLU) activation function.
    kTanh     ///< Hyperbolic tangent (tanh) activation function.
};

/**
 * @enum WeightInitFunctions
 * @brief An enumeration representing different weight initialization functions for neural networks.
 */
enum WeightInitFunctions {
    kNormal, ///< Normal weight initialization.
    kXavier, ///< Xavier weight initialization.
    kHE      ///< He weight initialization.
};

/**
 * @struct Const
 * @brief Contains constant values used in the application.
 */
struct Const {
    constexpr static size_t letter_width = 28;           ///< Width of letters in pixels.
    constexpr static size_t letter_height = 28;          ///< Height of letters in pixels.
    constexpr static size_t max_layer = letter_width * letter_height; ///< Maximum number of layers.
    constexpr static size_t min_layer = 26;             ///< Minimum number of layers.
    inline static std::pair<fp_type, fp_type> target = { 0.0, 1.0 }; ///< Default target values.
    inline static int default_train_proportion = 10;      ///< Default training data proportion (out of 10).
    inline static int error_updates = 10;                ///< Number of error updates.
};

/**
 * @struct Func
 * @brief Contains various utility functions used in the application.
 */
struct Func {
    // inline static fp_type xavier = 2.0; ///< Xavier initialization factor.

    /**
     * @brief Initialize weights using the Xavier initialization method.
     *
     * @param rows The number of rows in the weight matrix.
     * @param cols The number of columns in the weight matrix.
     * @return Initialized weight value.
     */
    static fp_type XavierWeightsInit(int rows, int cols);
    
    static fp_type HEWeightsInit(int rows, int cols);

    /**
     * @brief Initialize weights using the normal distribution with specified mean and standard deviation.
     *
     * @param mean The mean of the normal distribution.
     * @param sd The standard deviation of the normal distribution.
     * @return Initialized weight value.
     */
    static fp_type NormalWeightsInit(fp_type mean, fp_type sd);

    /**
     * @brief Calculate the ReLU (Rectified Linear Unit) activation function.
     *
     * @param x Input value.
     * @return Result of the ReLU activation.
     */
    static fp_type ActivationReLU(const fp_type x);

    /**
     * @brief Calculate the derivative of the ReLU (Rectified Linear Unit) activation function.
     *
     * @param x Input value.
     * @return Derivative of the ReLU activation.
     */
    static fp_type DerivativeActivationReLU(const fp_type x);

    /**
     * @brief Calculate the sigmoid activation function.
     *
     * @param x Input value.
     * @return Result of the sigmoid activation.
     */
    static fp_type ActivationSigmoid(const fp_type x);

    /**
     * @brief Calculate the derivative of the sigmoid activation function.
     *
     * @param x Input value.
     * @return Derivative of the sigmoid activation.
     */
    static fp_type DerivativeActivationSigmoid(const fp_type x);

    /**
     * @brief Calculate the SiLU (Sigmoid Linear Unit) activation function.
     *
     * @param x Input value.
     * @return Result of the SiLU activation.
     */
    static fp_type ActivationSiLU(const fp_type x);

    /**
     * @brief Calculate the derivative of the SiLU (Sigmoid Linear Unit) activation function.
     *
     * @param x Input value.
     * @return Derivative of the SiLU activation.
     */
    static fp_type DerivativeActivationSiLU(const fp_type x);

    /**
     * @brief Calculate the mean error from a vector of error values.
     *
     * @param error Vector of error values.
     * @return Mean error value.
     */
    static fp_type MeanError(const std::vector<fp_type> &error);


    static fp_type ActivationTanh(const fp_type x);
    static fp_type DerivativeActivationTanh(const fp_type x);

    static std::vector<fp_type> Softmax(const std::vector<fp_type> &logits);
};


} // namespace s21