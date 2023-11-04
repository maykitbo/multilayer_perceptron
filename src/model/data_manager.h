#pragma once

#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

#include "../common/strategy.h"
#include "../common/types.h"

namespace s21 {



/**
 * @enum LetterRotate
 * @brief Enum for letter rotation
 * @var LetterRotate::kNoRotate
 * No rotation
 * @var LetterRotate::k90Rotate
 * 90 degrees rotation
 */
enum LetterRotate {
    kNoRotate,
    k90Rotate
};


/**
 * @class DataManager
 * @brief Class for managing and processing data
 */
class DataManager {
    public:
        /**
         * @brief Construct a new Data Manager object.
         * @param file_path Path to the file with data.
         * @param bias Bias for the letter name (default is 0).
         * @param rotate Letter rotation option (default is kNoRotate).
         * @param width Letter width (default is Const::letter_width).
         * @param height Letter height (default is Const::letter_height).
         * @param classes Number of classes (default is Const::min_layer).
         */
        DataManager(const std::string &file_path, int bias = 0, LetterRotate rotate = kNoRotate,
                    size_t width = Const::letter_width, size_t height = Const::letter_height, unsigned classes = Const::min_layer);

        /**
         * @brief default Construct a new Data Manager object
         */
        DataManager() = default;

        /**
         * @brief Shuffle the dataset
         */
        void Shuffle();

        /**
         * @brief Split the dataset into train and test
         * test dataset is (1 / proportion) of the whole dataset
         * train dataset is (1 - 1 / proportion) of the whole dataset
         * @param proportion proportion of the test dataset
         */
        void Split(unsigned proportion);

        // /**
        //  * @brief Sets the metric for the dataset manager.
        //  * if SetMetrics have been called, dataset will process more often to the letters with the worst metrics
        //  * @param metric A vector of floating-point values representing the metric for evaluating model predictions.
        //  * The metrics coulb be precision, recall, F1, etc.
        //  */
        // void SetMetric(const std::vector<fp_type> &metric);

        /**
         * @brief passes throw the test part of dataset and applies the function to each letter
         * @param func A user-defined function that processes each letter in the testing dataset.
         */
        void ForTest(const std::function<void(data_t&)> func);

        /**
         * @brief passes throw the train part of dataset and applies the function to each letter
         * if SetMetrics have been called, then the function will be applied more often to the letters with the worst metrics
         * @param func A user-defined function that processes each letter in the training dataset.
         */
        void ForTrain(const std::function<void(data_t&)> func, int batch);
        
        /**
         * @brief change part of train dataset. Creates new part of train dataset from test dataset
         */
        void CrossUpdate();

        /**
         * @brief Get the Size of the dataset
         * @return unsigned int size of the dataset
         */
        unsigned int Size() { return size_; }

        /**
         * @brief Get the Size of the test dataset
         * @return unsigned int size of the test dataset
         */
        unsigned int TrainSize() { return size_ * train_proportion_; }

        /**
         * @brief throw an error if the dataset is of a different size
         */
        void Validate(size_t letter_size, unsigned classes);

        /**
         * @brief Prints the letter to the console
         * @param one the letter to be printed
         */
        static void PrintLetter(const data_t &one);

        /**
         * @brief Prints n letters to the console
         * @param n the number of letters to be printed
         */
        void Printn(int n);

        /**
         * @brief Get the Letters object
         * @return dataset_t& the dataset
         */
        dataset_t &GetLetters() { return letters_; }

        /**
         * @brief Get the Letters object
         * @return const dataset_t& the dataset
         */
        const dataset_t &GetLetters() const { return letters_; }

        void IncreaseRoatation(int angle);
    
    private:
        void ReadNoRotate(std::fstream &file, data_vector &letter);
        void Read90Rotate(std::fstream &file, data_vector &letter);
        // void Read180Rotate(std::fstream &file, data_vector &letter);
        // void Read270Rotate(std::fstream &file, data_vector &letter);
        auto ReadFunctionSwitch(LetterRotate rotate);

        std::vector<fp_type> Rotate(const std::vector<fp_type> &source, int angle);

        dataset_t letters_;
        size_t width_, height_, size_ = 0;
        fp_type test_proportion_, train_proportion_;
        // std::vector<fp_type> metric_;
        // fp_type metric_sum_;
        unsigned cross_k_, classes_;

};

} // namespace s21