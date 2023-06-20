#pragma once

#include "../common/strategy.h"
#include "../common/types.h"
#include "../lib/random.h"
#include "../lib/m_sstr.h"
#include "model.h"
#include "../lib/m_thread.h"

#include <limits>
#include <vector>

template<class T>
class MatrixModel;

template<class T>
class MatrixLayer {
    friend MatrixModel<T>;
    using matrix = typename TypesT<T>::matrix;
    using arith = typename TypesT<T>::arith;

    public:
        MatrixLayer(size_t rows, size_t cols, ThreadManager &thm);
        // MatrixLayer(std::ifstream &file);
        MatrixLayer() = delete;

        void Signal(const std::vector<T> *source);

        void UpdateWeights(const std::vector<T> *source);
        void Update(const std::vector<T> *source);
        void Update(MatrixLayer<T> &prev_layer);
        void UpdateGradientsBiases();
        void UpdateError(const std::vector<T> &target);

        void PrintInfo();
        void PrintNeurons();
        // ~MatrixLayer() {
        //     std::cout << "MatrixLayer destructor\n";
        // }

    private:
        matrix weights_, delta_weights_;
        std::vector<T> biases_, destination_, gradients_, error_;
        inline static T learning_rate_;
        ThreadManager &thm_;
};

template<class T>
class MatrixModel : public Model<T> {
    using letter = typename TypesT<T>::letter;
    using matrix = typename TypesT<T>::matrix;

    public:
        MatrixModel(const Types::vector_s &layer_sizes, T learning_rate = 1);
        MatrixModel(const std::string &file_name);
        MatrixModel() = delete;

        void Forward() override;

        void Backward() override;

        int GetResult() override;

        void Print();
        void ToFile(const std::string &file_name);

        // ~MatrixModel() {
        //     std::cout << "MatrixModel destructor\n";
        // }
    
    private:
        std::vector<MatrixLayer<T>> layers_;
        std::vector<T> target_output_;
        ThreadManager thm_;
        std::vector<T> *source_;
};

/////////////////////////////////////////////////////////////////////
// CONSTRUCTORS

template<class T>
MatrixLayer<T>::MatrixLayer(size_t rows, size_t cols, ThreadManager &thm) :
        weights_(rows, cols, [] {
            T res = 0.0;
            while(res == 0.0) {
                res = s21::Random::Normal<T>(ConstT<T>::w_mean, ConstT<T>::w_sd);
            }
            return res;
        }),
        // weights_(rows, cols, [] { return s21::Random::Normal<T>(ConstT<T>::w_mean, ConstT<T>::w_sd); }),
        delta_weights_(rows, cols, 0), destination_(cols), biases_(cols), gradients_(cols), thm_(thm), error_(cols) {}

template<class T>
MatrixModel<T>::MatrixModel(const Types::vector_s &layer_sizes, T learning_rate) :
        target_output_(layer_sizes.back(), ConstT<T>::target.first), thm_(4) {

    MatrixLayer<T>::learning_rate_ = learning_rate;
    for (int k = 0; k < layer_sizes.size() - 1; ++k) {
        layers_.emplace_back(layer_sizes[k], layer_sizes[k + 1], thm_);
    }
}

/////////////////////////////////////////////////////////////////////////
// FORWARD

template<class T>
void MatrixLayer<T>::Signal(const std::vector<T> *source) {
    arith::Mul(*source, weights_, destination_);

    for (size_t g = 0; g < weights_.GetCols(); ++g) {
        destination_[g] = Func<T>::Sigmoid(destination_[g] + biases_[g]);
    }
}

template<class T>
void MatrixModel<T>::Forward() {
    auto layer = layers_.begin();

    source_ = &Model<T>::letter_->second;

    layer->Signal(source_);
    auto cur = &layer->destination_;
    for (++layer; layer != layers_.end(); ++layer) {
        layer->Signal(cur);
        cur = &layer->destination_;
    }
}

////////////////////////////////////////////////////////////////////////
// BACKWARD

template<class T>
void MatrixLayer<T>::UpdateWeights(const std::vector<T> *source) {
    thm_.LoopExecute(weights_.GetRows(), [&] (int k) {
        for (int g = 0; g < weights_.GetCols(); ++g) {
            // delta_weights_(k, g) = ConstT<T>::momentum * delta_weights_(k, g) + learning_rate_ * gradients_[g] * (*source)[k];
            // weights_(k, g) += delta_weights_(k, g);
            weights_(k, g) += learning_rate_ * gradients_[g] * (*source)[k];
        }
    });
}

template<class T>
void MatrixLayer<T>::UpdateError(const std::vector<T> &target) {
    for (int g = 0; g < error_.size(); ++g) {
        error_[g] = target[g] - destination_[g];
    }
}

template<class T>
void MatrixLayer<T>::UpdateGradientsBiases() {
    for (int g = 0; g < destination_.size(); ++g) {
        gradients_[g] = Func<T>::DerivativeSigmoid(destination_[g]) * error_[g];
        biases_[g] += learning_rate_ * error_[g];
    }
}

template<class T>
void MatrixLayer<T>::Update(const std::vector<T> *source) {
    UpdateGradientsBiases();
    UpdateWeights(source);
}

template<class T>
void MatrixLayer<T>::Update(MatrixLayer<T> &prev_layer) {
    Update(&prev_layer.destination_);
    arith::MulBT(gradients_, weights_, prev_layer.error_);
}

template<class T>
void MatrixModel<T>::Backward() {
    target_output_[Model<T>::letter_->first] = ConstT<T>::target.second;

    int layer_k = layers_.size() - 1;

    layers_[layer_k].UpdateError(target_output_);

    for ( ; layer_k >= 1; --layer_k) {
        layers_[layer_k].Update(layers_[layer_k - 1]);
    }

    layers_[0].Update(source_);

    target_output_[Model<T>::letter_->first] = ConstT<T>::target.first;
}

//////////////////////////////////////////////////////////////////////////
// OTHER

template<class T>
int MatrixModel<T>::GetResult() {
    T max = -std::numeric_limits<T>::infinity(), result = 0.0;
    const auto &output = layers_.back().destination_;
    for (int k = 0; k < output.size(); ++k) {
        if (max < output[k]) {
            max = output[k];
            result = k;
        }
    }
    // s21::SStr::Print(output);
    return result;
}



template<class T>
MatrixModel<T>::MatrixModel(const std::string &file_name) {
    std::ifstream file(file_name);
    int size;
    file >> size;
    file >> MatrixLayer<T>::learning_rate_;
    for (int k = 0; k < size; ++k) {
        layers_.emplace_back(file);
    }
    file.close();
}

template<class T>
void MatrixModel<T>::ToFile(const std::string &file_name) {
    std::ofstream file(file_name);
    file << layers_.size() << ' ' << MatrixLayer<T>::learning_rate_ << '\n';
    for (const auto &layer : layers_) {
        file << layer.weights_ << layer.biases_;
    }
    file.close();
}

template<class T>
void MatrixModel<T>::Print() {
    for (auto &i : layers_) {
        i.PrintInfo();
    }
    layers_.back().PrintNeurons();

}

template<class T>
void MatrixLayer<T>::PrintInfo() {
    // std::cout << weights_.GetCols() << ":" << weights_.GetRows() << " " << destination_.GetCols() << " " << biases_.GetCols() << "\n";
}

template<class T>
void MatrixLayer<T>::PrintNeurons() {
    // for (auto &i : destination_.ToVector()) {
    //     printf("%.8f ", i);
    // }
    // std::cout << "\n";
}