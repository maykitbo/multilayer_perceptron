#pragma once

#include <vector>
#include "data_manager.h"
#include "../common/metrics.h"
#include "../common/perceptron_settings.h"
#include "../lib/m_time.h"


#include "../common/error_thread.h"

namespace s21 {

/**
 * @class Model
 * @brief Abstract class for perceptron model
 */
class Model {
    public:
        /**
         * @brief Construct a new Model object
         */
        Model() {}

        /**
         * @brief Train the model using training data.
         * @param inputs DataManager dataset.
         * @param epoch_count Number of training epochs.
         * @param batch Batch size for training.
         * @param stop A flag to control the training loop.
         * @param error_thread Flag to send errors to a separate thread.
         * @param metric_thread Flag to send metrics to a separate thread.
         */
        void Train(DataManager &inputs, unsigned int epoch_count, unsigned batch,
                bool &stop, bool error_thread = true, bool metric_thread = true);

        /**
         * @brief Test the model using test data.
         * @param inputs DataManager dataset.
         * @return Metrics object containing evaluation results.
         */
        Metrics Test(DataManager &inputs);

        /**
         * @brief Predict the letter for a given data vector.
         * @param letter Data vector for prediction.
         * @return Predicted letter as a char.
         */
        char Predict(data_vector &input);

        /**
         * @brief virtual Forward function. Forward the letter through the model
         * have to be implemented in the child class
         */
        virtual void Forward(data_vector &input) = 0;

        /**
         * @brief virtual Backward function. Backward the letter through the model
         * have to be implemented in the child class
         * @param answer int Answer. Correct class of the letter
         */
        virtual void Backward(data_t &input) = 0;

        /**
         * @brief virtual function to update the weights of the model
         * have to be implemented in the child class
         */
        virtual void UpdateWeights() = 0;

        /**
         * @brief Virtual function to retrieve the model's result.
         * @return Class prediction for the letter.
         */
        virtual int GetResult() = 0;

        /**
         * @brief Virtual function to save the model to a file.
         * @param filename File name for saving the model.
         */
        virtual void ToFile(const std::string &filename) = 0;

        /**
         * @brief Set a callback function for error reporting (if enabled).
         * @param func Callback function taking mean error and epoch number.
         */
        void SetErrorThread(const std::function<void(fp_type, unsigned)> &func);

        /**
         * @brief Set a callback function for metric reporting (if enabled).
         * @param func Callback function taking Metrics as an argument.
         */
        void SetMetricThread(const std::function<void(Metrics)> &func);

        /**
         * @brief Default virtual destructor
         */
        virtual ~Model() = default;

    protected:
        // std::vector<const data_vector&> sources_;

        /**
         * @brief Settings for the Perceptron model.
         */
        PerceptronSettings settings_;

        /**
         * @brief Error information for the model.
         */
        Error error_;

        /**
         * @brief Callback function for reporting metrics.
         * This function is called when metrics are available.
         */
        std::function<void(Metrics)> metrics_func_;

        /**
         * @brief This function must be overridden in derived classes.
         * @return Mean error for the model.
         */
        virtual fp_type GetMeanError() = 0;

        /**
         * @brief Virtual method to update the learning rate every epoch (can be overridden).
         * This method can be overridden in derived classes to implement custom
         * learning rate update logic.
         */
        virtual void UpdateLR() {}

    private:
        void EpochTest(DataManager &letters, int64_t training_time);
};


/**
 * @brief A template class for performing cross-validation with machine learning models.
 *
 * @tparam ModelType The type of the machine learning model to be used.
 */
template<class ModelType>
struct CrossValidation {
    /**
     * @brief Run cross-validation on a dataset using the specified machine learning model.
     *
     * @param letters DataManager dataset.
     * @param settings PerceptronSettings specifying model configuration.
     * @param cross The number of cross-validation folds.
     * @param epoch_count Number of training epochs for each fold.
     * @param stop A flag to control the training loop.
     * @param matrics_thread A callback function to handle metrics reporting.
     * @param error_thread A callback function to handle error reporting.
     */
    static void Run(DataManager &letters, const PerceptronSettings &settings, unsigned int cross, unsigned int epoch_count, bool &stop,
            std::function<void(Metrics&)> matrics_thread, std::function<void(fp_type, unsigned int)> error_thread);
};


template<class ModelType>
void CrossValidation<ModelType>::Run(DataManager &letters, const PerceptronSettings &settings, unsigned int cross, unsigned int epoch_count, bool &stop,
            std::function<void(Metrics&)> matrics_thread, std::function<void(fp_type, unsigned int)> error_thread) {
    
    // Validate dataset and perform cross-validation split
    letters.Validate(settings.layers.front(), settings.layers.back());
    letters.Split(cross);

    for (unsigned int k = 0; k < cross; ++k) {
        // Create a new model instance for each fold
        ModelType model(settings);
        model.SetErrorThread(error_thread);

        // Train the model and perform testing
        model.Train(letters, epoch_count, stop, true, false);
        auto m = model.Test(letters);
        matrics_thread(m);

        // Update dataset for the next fold
        letters.CrossUpdate();
    }
}

} // namespace s21
