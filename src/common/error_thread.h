#pragma once

#include "strategy.h"


namespace s21 {

/**
 * @class Error
 * @brief Represents a class for collecting and reporting error information during training.
 */
class Error {
    public:
        /**
         * @brief Set a callback function to handle error reporting.
         *
         * @param f Callback function taking error value and epoch number as arguments.
         */
        void SetFunc(const std::function<void(fp_type, unsigned int)> &f);

        /**
         * @brief Set the current training epoch number.
         *
         * @param epoch The current epoch number.
         */
        void SetEpoch(unsigned int epoch);
        
        /**
         * @brief Collect error data.
         *
         * @param error The error value to collect.
         */
        void Collect(fp_type error);

        /**
         * @brief Set the size of the dataset.
         *
         * @param size The size of the dataset.
         */
        void SetDatasetSize(unsigned int size);

        /**
         * @brief Clear collected error data.
         */
        void Clear();

        /**
         * @brief Send collected error data to the callback function (if set).
         */
        void Send();
    
    private:
        fp_type value_;
        unsigned int epoch_;
        unsigned int count_ = 0;
        std::function<void(fp_type, unsigned int)> func_;
        unsigned int period_;     
};

} // namespace s21
