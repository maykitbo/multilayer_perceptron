#pragma once

#include <vector>
#include "strategy.h"

#include "types.h"

namespace s21 {

/**
 * @struct MappedLettersMetrics
 * @brief Contains mapped metrics for a set of classes, including precision, recall, F1 score, accuracy,
 *        and training/testing times.
 */
struct MappedLettersMetrics {
    std::vector<std::pair<char, fp_type>> precision_map; ///< Precision values for each class.
    std::vector<std::pair<char, fp_type>> recall_map;    ///< Recall values for each class.
    std::vector<std::pair<char, fp_type>> F1_map;        ///< F1 score values for each class.
    fp_type accuracy = 0;                                ///< Overall accuracy. Default value is 0.0.
    int64_t training_time = 0;                           ///< Time taken for training. Default value is 0.
    int64_t testing_time = 0;                            ///< Time taken for testing. Default value is 0.

    /**
     * @brief Constructs MappedLettersMetrics with the specified number of classes (size).
     *
     * @param size The number of classes.
     */
    MappedLettersMetrics(unsigned int size) :
            precision_map(size), recall_map(size), F1_map(size) {}
    
    /**
     * @brief Default constructor for MappedLettersMetrics.
     */
    MappedLettersMetrics() = default;
};


/**
 * @struct Metrics
 * @brief Contains metrics such as precision, recall, F1 score, accuracy, and training/testing times.
 */
struct Metrics {
    std::vector<fp_type> precision;   ///< Precision values for each class.
    std::vector<fp_type> recall;      ///< Recall values for each class.
    std::vector<fp_type> F1;          ///< F1 score values for each class.
    fp_type accuracy = 0.0;           ///< Overall accuracy. Default value is 0.0.
    int64_t training_time = 0;        ///< Time taken for training. Default value is 0.
    int64_t testing_time = 0;         ///< Time taken for testing. Default value is 0.

    /**
     * @brief Constructs Metrics with the specified number of classes (size).
     *
     * @param size The number of classes.
     */
    Metrics(unsigned int size);

    /**
     * @brief Default constructor for Metrics.
     */
    Metrics() = default;

    /**
     * @brief Get mapped metrics suitable for each class.
     *
     * @return MappedLettersMetrics containing mapped precision, recall, F1, accuracy, and times.
     */
    MappedLettersMetrics GetMappedLettersMetrics();
};


/**
 * @class MetricsMaker
 * @brief A utility class for creating and updating metrics, including precision, recall, F1 score, accuracy,
 *        and training/testing times.
 */
class MetricsMaker {
    public:
        /**
         * @brief Constructs a MetricsMaker with the specified number of classes (size).
         *
         * @param size The number of classes.
         */
        MetricsMaker(unsigned int size);

        /**
         * @brief Update metrics based on the provided result and expected value.
         *
         * @param result The predicted class label.
         * @param expected The expected class label.
         */
        void operator()(int result, int expected);

        /**
         * @brief Get access to the computed metrics.
         *
         * @return A reference to the Metrics object containing computed metrics.
         */
        Metrics &operator()();

        /**
         * @brief Clear the internal state and reset metrics.
         */
        void Clear();

    private:
        Metrics metrics_;
        std::vector<std::pair<int, int>> precision_helper_;
        std::vector<std::pair<int, int>> recall_helper_;
        unsigned int size_;
        int count_ = 0, accuracy_helper_ = 0;
};

} // namespace s21
