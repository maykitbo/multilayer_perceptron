#pragma once

#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

#include "../common/strategy.h"
#include "../common/types.h"

namespace s21 {

// struct Parser {
//     static void CSV() {

//     }
// }

class DataManager {
    dataset_t letters_;
    int size_ = 0, cross_k_;
    fp_type test_proportion_ = 1.0, train_proportion_ = 0.0;
    std::vector<fp_type> metric_;
    fp_type metric_sum_;
    fp_type d_angel_;
    const std::function<void(std::fstream&, data_vector&)> &read_function_;

    static void PrintLetter(const data_vector &one);
    void Variation0(const data_vector &sample, int name);
    void Variation1(const data_vector &sample, int name);
    void Variation2(const data_vector &sample, int name);

    public:

        // angel = d_angel * 90
        DataManager(const std::string &file_path, int bias = 0, int d_angel = 0);
        DataManager() = default;
        void Shuffle();
        void Print() const;
        void ForTest(const std::function<void(data_vector&, int)> func);
        void ForTrain(const std::function<void(data_vector&, int)> func);
        
        void SetMetric(const std::vector<fp_type> &metric);
        void Split(int proportion);
        void Expand();
        void CrossUpdate();
        int Size() { return size_; }

        
        static void ReadRotate(std::fstream &file, data_vector &letter) {
            for (int k = 0; k < Const::letter_height; ++k) {
                for (int g = 0; g < Const::letter_width; ++g) {
                    file >> letter[g * Const::letter_height + k];
                    letter[g * Const::letter_height + k] /= 256.0;
                    file.ignore();
                }
            }
        }

        static void Read(std::fstream &file, data_vector &letter) {
            for (auto &i : letter) {
                file >> i;
                i /= 256.0;
                file.ignore();
            }
        }
};

DataManager::DataManager(const std::string &file_path, int bias, int d_angel) :
        letters_(Const::min_layer), d_angel_(d_angel),
        read_function_(d_angel == 0.0 ? Read : ReadRotate) {
    std::fstream file(file_path);
    while (file) {
        int name;
        file >> name;
        name += bias;
        file.ignore();
        letters_[name].emplace_back(Const::max_layer);
        read_function_(file, letters_[name].back());
        ++size_;
    }
    file.close();

    // for (int k = 0; k < 10; ++k) {
    //     int name = Random::Int<int>(0, Const::min_layer - 1);
    //     PrintLetter(letters_[name][Random::Int<int>(0, letters_[name].size() * train_proportion_)]);
    // }
}

void DataManager::ForTest(const std::function<void(data_vector&, int)> func) {
    for (int name = 0; name < Const::min_layer; ++name) {
        for (auto iter = letters_[name].end() - letters_[name].size() * test_proportion_;
                iter != letters_[name].end(); ++iter) {
            func(*iter, name);
        }
    }
}

void DataManager::ForTrain(const std::function<void(data_vector&, int)> func) {
    std::vector<bool> trained(Const::min_layer, false);

    // std::vector<int> counts(Const::min_layer, 0);

    for (int k = 0; k < size_ * train_proportion_; ++k) {
        fp_type selection = s21::Random::Uniform<fp_type>(0.0, metric_sum_);
        for (int name = 0; name < Const::min_layer; ++name) {
            selection -= metric_[name];
            if (selection <= 0.0) {
                trained[name] = true;

                // ++counts[name];

                func(letters_[name][Random::Int<int>(0, letters_[name].size() * train_proportion_)], name);
                break;
            }
        }
        // int name = Random::Int<int>(0, Const::min_layer - 1);
        // func(letters_[name][Random::Int<int>(0, letters_[name].size() * train_proportion_)], name);
    }
    for (int name = 0; name < Const::min_layer; ++name) {

        // std::cout << counts[name] << ' ';

        if (!trained[name]) {
            func(letters_[name][Random::Int<int>(0, letters_[name].size() * train_proportion_)], name);
        }
    }

    std::cout << '\n';
}

void DataManager::Split(int proportion) {
    metric_ = std::vector<fp_type>(Const::min_layer, 1.0);
    metric_sum_ = Const::min_layer;
    test_proportion_ = 1.0 / (fp_type)proportion;
    train_proportion_ = 1.0 - test_proportion_;
    cross_k_ = proportion - 1;
}

void DataManager::SetMetric(const std::vector<fp_type> &metric) {
    metric_sum_ = 0.0;
    for (int k = 0; k < Const::min_layer; ++k) {
        metric_[k] = 1.0 / metric[k];
        metric_sum_ += metric_[k];
    }
}

void DataManager::Shuffle() {
    for (auto iter = letters_.begin(); iter != letters_.end(); ++iter) {
        std::shuffle(iter->begin(), iter->end(),
            std::default_random_engine(std::random_device{}()));
    }
}

void DataManager::CrossUpdate() {
    if (cross_k_ == 0) {
        std::cout << "cros_k == 0\n";
        return;
    }
    for (int name = 0; name < Const::min_layer; ++name) {
        int end = cross_k_ * letters_[name].size() * test_proportion_;
        int start = (cross_k_ - 1) * letters_[name].size() * test_proportion_;
        std::rotate(letters_[name].begin() + start, letters_[name].begin() + end, letters_[name].end());
    }
    --cross_k_;
}

void DataManager::Expand() {
    // int stop = 0;
    for (int name = 0; name < Const::min_layer; ++name) {
        int size = letters_[name].size();
        for (int k = 0; k < size; k += 1) {
            Variation0(letters_[name][k], name);
            // Variation1(letters_[name][k], name);
            // Variation2(letters_[name][k - 1], name);
        }
    }
}

void DataManager::Variation1(const data_vector &sample, int name) {
    auto &new_letter = letters_[name].emplace_back(Const::max_layer, 0.0);
    for (int k = 0; k < Const::letter_height; ++k) {
        for (int g = 0; g < Const::letter_width; ++g) {
            new_letter[k * Const::letter_height + g] = std::sqrt(sample[k * Const::letter_height + g]);
        }
    }
    ++size_;
}

void DataManager::Variation2(const data_vector &sample, int name) {
    auto &new_letter = letters_[name].emplace_back(Const::max_layer, 0.0);
    for (int k = 0; k < Const::letter_height; ++k) {
        for (int g = 0; g < Const::letter_width; ++g) {
            new_letter[k * Const::letter_height + g] = std::pow(sample[k * Const::letter_height + g], 2.0);
        }
    }
    ++size_;
}

void DataManager::Variation0(const data_vector &sample, int name) {
    auto &new_letter = letters_[name].emplace_back(Const::max_layer, 0.0);
    int delta_g = 0, delta_k = 0;
    while (delta_g + delta_k == 0) {
        delta_g = Random::Int(-2, 2);
        delta_k = Random::Int(-2, 2);
    }
    int k_bias = std::max(delta_k, 0), g_bias = std::max(delta_g, 0);
    for (int k = k_bias; k < Const::letter_height - k_bias; ++k) {
        for (int g = g_bias; g < Const::letter_width - g_bias; ++g) {
            new_letter[k * Const::letter_height + g] = sample[(k + delta_k) * Const::letter_height + g + delta_g];
        }
    }
    ++size_;
}

void DataManager::Print() const {
//     int c = 0;
//     for (auto i = letters_.begin(); i != letters_.end() && c < 3; ++i, ++c) {
//         printf("%c:\n", i->first + 'a');
//       void Expand();  auto iter = i->second.begin();
//         for (int k = Const::letter_height - 1; k != 0; --k) {
//             for (int g = 0; g < Const::letter_width; ++g) {
//                 if (*iter > 0.1)
//                     printf("\033[0;31m");
//                 else
//                     printf("\033[0m");
//                 printf("%2.2f", *iter);
//                 ++iter;
//             }
//             std::cout << "\n";
//         }
//         std::cout << "\n";
//     }
}

void DataManager::PrintLetter(const data_vector &one) {
    auto iter = one.begin();
    for (int k = Const::letter_height - 1; k != -1; --k) {
        for (int g = 0; g < Const::letter_width; ++g) {
            if (*iter > 0.2)
                printf("\033[0;31m");
            else
                printf("\033[0m");
            printf("%2.2f", *iter);
            ++iter;
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

} // namespace s21
