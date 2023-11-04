#pragma once

#include "../common/perceptron_settings.h"
#include "../lib/random.h"
#include "../lib/m_sstr.h"
#include "model.h"
#include <numeric>
#include <limits>
#include <algorithm>
#include <execution>
#include <thread>

#include <cblas.h>

namespace s21 {

class MatrixModel;

/**
 * @class MatrixLayer
 * @brief Represents a layer in a matrix-based neural network model.
 */
class MatrixLayer {
    friend MatrixModel;

    public:
        /**
         * @brief Constructs a MatrixLayer with the specified number of rows and columns, learning rate, and settings.
         *
         * @param rows The number of rows in the layer's weight matrix.
         * @param cols The number of columns in the layer's weight matrix.
         * @param lr The learning rate for weight updates.
         * @param settings PerceptronSettings containing layer configuration.
         */
        MatrixLayer(size_t rows, size_t cols, fp_type lr, const PerceptronSettings &settings);

        /**
         * @brief Constructs a MatrixLayer with the specified number of rows and columns, learning rate, settings,
         *        and initializes weights and biases from a file.
         *
         * @param rows The number of rows in the layer's weight matrix.
         * @param cols The number of columns in the layer's weight matrix.
         * @param lr The learning rate for weight updates.
         * @param settings PerceptronSettings containing layer configuration.
         * @param file An input file stream for weight and bias initialization.
         */
        MatrixLayer(size_t rows, size_t cols, fp_type lr, const PerceptronSettings &settings, std::ifstream &file);

        MatrixLayer() = delete;
        ~MatrixLayer() = default;

        /**
         * @brief Signals the layer with input data.
         * forward pass
         *
         * @param source Pointer to the source input data.
         */
        void Signal(const std::vector<fp_type> &source);

        void OutputSignal(const std::vector<fp_type> &source);

        /**
         * @brief Updates the weights based on the source data.
         * backward pass
         *
         * @param source Pointer to the source input data.
         */
        void UpdateGradientsMatrix(const std::vector<fp_type> &source);


        void UpdateWeights();
        void UpdateWeights(int threads);


        /**
         * @brief Updates the first layer's weights based on the source data.
         * backward pass
         *
         * @param source Pointer to the source input data.
         */
        // void BackwardFirst(const std::vector<fp_type> &source);

        void OutputBackward(MatrixLayer &prev_layer, int answer);
        void InputBackward(const std::vector<fp_type> &source);
        void HiddenBackward(MatrixLayer &prev_layer);

        /**
         * @brief Updates the weights based on the previous layer's output.
         * backward pass
         *
         * @param prev_layer Reference to the previous layer.
         */
        void Backward(MatrixLayer &prev_layer);

        // void Backward1(MatrixLayer &prev_layer);


        /**
         * @brief Updates gradients and biases for backpropagation.
         */
        void UpdateGradients();

        /**
         * @brief Updates the error for backpropagation based on the target data.
         *
         * @param target The target output data.
         */
        // void UpdateError(int answer);

    private:
        matrix_t weights_, delta_weights_;
        matrix_t gradients_matrix_;
        std::vector<fp_type> biases_, destination_;
        std::vector<fp_type> gradients_, gradients_sum_;
        const PerceptronSettings &settings_;
        fp_type learning_rate_;

        matrix_t adam_cache_;
        matrix_t adam_m_;
        matrix_t adam_v_;
        size_t adam_t_{1};
        std::vector<float> delta_biases_;
};


/**
 * @class MatrixModel
 * @brief Represents a matrix-based neural network model.
 */
class MatrixModel : public Model {

    public:
        /**
         * @brief Constructs a MatrixModel with the specified settings.
         *
         * @param settings PerceptronSettings containing model configuration.
         */
        MatrixModel(const PerceptronSettings &settings);

        PerceptronSettings &GetSettings() { return settings_; }

        /**
         * @brief Constructs a MatrixModel by loading model data from a file.
         *
         * @param file_name The name of the file containing model data.
         */
        MatrixModel(const std::string &file_name);

        MatrixModel() = delete;

        /**
         * @brief Overrides the forward propagation method for the matrix-based model.
         */
        void Forward(data_vector &input) override;

        /**
         * @brief Overrides the backward propagation method for the matrix-based model.
         *
         * @param answer The correct class label for the input data.
         */
        void Backward(data_t &input) override;

        /**
         * @brief Overrides the method to update the weights for the model.
         */
        void UpdateWeights() override;

        ~MatrixModel() = default;

        /**
         * @brief Overrides the method to retrieve the model's result.
         *
         * @return The predicted class label.
         */
        int GetResult() override;

        /**
         * @brief Overrides the method to save the model to a file.
         *
         * @param file_name The name of the file to save the model.
         */
        void ToFile(const std::string &file_name) override;

        /**
         * @brief Overrides the method to update the learning rate for the model.
         */
        void UpdateLR() override;
    
    private:
        std::vector<MatrixLayer> layers_;

        fp_type GetMeanError() override {
            // return Func::MeanError(layers_.back().error_);
            return Func::MeanError(layers_.back().gradients_);
        }

};

} // namespace s21