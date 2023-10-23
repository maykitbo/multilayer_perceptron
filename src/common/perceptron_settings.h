#pragma once

#include "strategy.h"
#include <sstream>

namespace s21 {

class MatrixLayer;
class GraphModel;

/**
 * @enum ModelType
 * @brief An enumeration representing different types of machine learning models.
 */
enum ModelType
{
    kMatrix, ///< Matrix-based model.
    kGraph   ///< Graph-based model.
};


/**
 * @struct PerceptronSettings
 * @brief Contains configuration settings for a perceptron-based machine learning model.
 */
struct PerceptronSettings {
    public:
        /**
         * @brief Default constructor for PerceptronSettings.
         */
        PerceptronSettings() = default;

        size_vector layers;         ///< Vector of layer sizes in the model.
        fp_type learning_rate;      ///< Learning rate for weight updates.
        fp_type momentum = 0.0;     ///< Momentum coefficient for weight updates. Default value is 0.0.
        fp_type lr_epoch_k = 1.0;   ///< Learning rate scaling factor per epoch. Default value is 1.0.
        fp_type lr_layers_k = 1.0;  ///< Learning rate scaling factor per layer. Default value is 1.0.
        fp_type weight_mean = 0.0;  ///< Mean for weight initialization. Default value is 0.0.
        fp_type weight_sd = 1.0;    ///< Standard deviation for weight initialization. Default value is 1.0.
        
        /**
         * @brief Set the activation function and its derivative based on an enumeration value.
         *
         * @param index The index representing the activation function.
         */
        void SetActivation(ActivationFunctions index);

        /**
         * @brief Set the weight initialization function based on an enumeration value.
         *
         * @param index The index representing the weight initialization function.
         */
        void SetWeightInit(WeightInitFunctions index);

        /**
         * @brief Validate the configuration settings to ensure they are within valid ranges.
         * throws std::runtime_error
         */
        void Validate() const;
        
        /**
         * @brief Constructor for PerceptronSettings that loads settings from an input file stream.
         *
         * @param os The input file stream to read settings from.
         */
        PerceptronSettings(std::ifstream &os);

        /**
         * @brief Save the configuration settings to an output stream.
         *
         * @param os The output stream to write settings to.
         */
        void ToFile(std::ostream& os);
    
    private:
        friend class MatrixLayer;
        friend class GraphModel;
        friend class GraphLayer;
        std::function<fp_type(fp_type)> activation;               ///< Activation function.
        std::function<fp_type(fp_type)> derivative_activation;    ///< Derivative of the activation function.
        std::function<fp_type(int, int)> weight_init;             ///< Weight initialization function.
        ActivationFunctions activation_i;                          ///< Activation function index.
        WeightInitFunctions weight_init_i;                        ///< Weight initialization function index.
};

} // namespace s21
