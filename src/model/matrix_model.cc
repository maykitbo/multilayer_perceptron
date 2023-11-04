#include "matrix_model.h"

using namespace s21;

/////////////////////////////////////////////////////////////////////
// CONSTRUCTORS

MatrixLayer::MatrixLayer(size_t rows, size_t cols, fp_type lr, const PerceptronSettings &settings)
    : weights_(rows, cols, [&] { return settings.weight_init(rows, cols); })
    , delta_weights_(rows, cols, 0.0)
    , biases_(cols)
    , destination_(cols)
    , gradients_(cols)
    , gradients_sum_(cols)
    , gradients_matrix_(rows, cols, 0.0)
    , settings_(settings)
    , learning_rate_(lr)
    // , adam_cache_(weights_)
    , adam_cache_(rows, cols, 0.0)
    , adam_m_(rows, cols, 0.0)
    , adam_v_(rows, cols, 0.0)
    , delta_biases_(cols, 0.0)
{}

MatrixModel::MatrixModel(const PerceptronSettings &settings)
{
    settings.Validate();
    settings_ = settings;
    fp_type lr = settings_.learning_rate;
    for (unsigned k = 0; k < settings_.layers.size() - 1; ++k)
    {
        layers_.emplace_back(settings_.layers[k], settings_.layers[k + 1], lr, settings_);
        lr *= settings_.lr_layers_k;
    }
}

/////////////////////////////////////////////////////////////////////////
// FORWARD

void MatrixLayer::Signal(const std::vector<fp_type> &source)
{
    // destination_ = source * weights_ + biases_
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        1, weights_.GetCols(), source.size(), 1.0,
        source.data(), source.size(), weights_.Data(),
        weights_.GetCols(), 0.0, destination_.data(), weights_.GetCols());

    for (unsigned g = 0; g < weights_.GetCols(); ++g)
    {
        destination_[g] = settings_.activation(destination_[g] + biases_[g]);
    }
}

void MatrixLayer::OutputSignal(const std::vector<fp_type> &source)
{
    // destination_ = source * weights_ + biases_
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        1, weights_.GetCols(), source.size(), 1.0,
        source.data(), source.size(), weights_.Data(),
        weights_.GetCols(), 0.0, destination_.data(), weights_.GetCols());
    
    destination_ = Func::Softmax(destination_);
}

void MatrixModel::Forward(data_vector &input)
{
    layers_[0].Signal(input);
    auto cur = &layers_[0].destination_;
    for (int k = 1; k < layers_.size() - 1; ++k)
    {
        layers_[k].Signal(*cur);
        cur = &layers_[k].destination_;
    }
    layers_.back().OutputSignal(*cur);
    // layers_.back().Signal(*cur);
}



////////////////////////////////////////////////////////////////////////
// BACKWARD

void MatrixLayer::UpdateGradientsMatrix(const std::vector<fp_type> &source)
{
    for (unsigned int g = 0; g < destination_.size(); ++g)
    {
        gradients_sum_[g] += gradients_[g];
    }

    // gradients_matrix_ = source ^ T * gradients_
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //     source.size(), gradients_.size(), 1, learning_rate_,
    //     source.data(), 1, gradients_.data(), gradients_.size(), 1.0,
    //     gradients_matrix_.Data(), gradients_matrix_.GetCols());
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        source.size(), gradients_.size(), 1, 1.0,
        source.data(), 1, gradients_.data(), gradients_.size(), 1.0,
        gradients_matrix_.Data(), gradients_matrix_.GetCols());
}

void MatrixLayer::UpdateGradients()
{
    for (unsigned g = 0; g < destination_.size(); ++g)
    {
        gradients_[g] = settings_.derivative_activation(destination_[g]) * gradients_[g];
    }
}

void MatrixLayer::Backward(MatrixLayer &prev_layer)
{
    // prev_layer.gradients_ = gradients_ * weights_ ^ T
    // adam_cache_ = weights_ + delta_weights_;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        1, weights_.GetRows(), weights_.GetCols(), 1.0,
        gradients_.data(), weights_.GetCols(), weights_.Data(),
        weights_.GetCols(), 0.0, prev_layer.gradients_.data(), weights_.GetRows());

    UpdateGradientsMatrix(prev_layer.destination_);
}

void MatrixLayer::OutputBackward(MatrixLayer &prev_layer, int answer)
{   
    for (unsigned g = 0; g < destination_.size(); ++g)
    {
        gradients_[g] = destination_[g];
    }
    gradients_[answer] -= Const::target.second;
    Backward(prev_layer);
}

void MatrixLayer::InputBackward(const std::vector<fp_type> &source)
{
    UpdateGradients();
    UpdateGradientsMatrix(source);
}

void MatrixLayer::HiddenBackward(MatrixLayer &prev_layer)
{
    UpdateGradients();
    Backward(prev_layer);
}

void MatrixModel::Backward(data_t &input)
{
    unsigned layer_k = layers_.size() - 1;

    layers_[layer_k].OutputBackward(layers_[layer_k - 1], input.second);
    for (--layer_k; layer_k >= 1; --layer_k)
    {
        layers_[layer_k].HiddenBackward(layers_[layer_k - 1]);
    }
    layers_[0].InputBackward(input.first);
}

//////////////////////////////////////////////////////////////////////////
// UPDATE

void MatrixLayer::UpdateWeights()
{
    for (unsigned g = 0; g < destination_.size(); ++g)
    {
        delta_biases_[g] = delta_biases_[g] * settings_.momentum +
                        gradients_[g] * (1.0 - settings_.momentum);
        // biases_[g] += learning_rate_ * gradients_sum_[g];
        biases_[g] -= learning_rate_ * delta_biases_[g];
    }

    for (unsigned k = 0; k < weights_.GetRows(); ++k)
    {
        for (unsigned g = 0; g < weights_.GetCols(); ++g)
        {
            // MOMENTUM
            delta_weights_(k, g) = delta_weights_(k, g) * settings_.momentum +
                            gradients_matrix_(k, g) * (1.0 - settings_.momentum);
            weights_(k, g) -= learning_rate_ * delta_weights_(k, g);

            // NESTEROV
            // delta_weights_(k, g) = delta_weights_(k, g) * settings_.momentum +
            //                 gradients_matrix_(k, g) * learning_rate_;
            // weights_(k, g) += delta_weights_(k, g);
            // adam_cache_(k, g) = weights_(k, g) + delta_weights_(k, g);

            // ADAM
            // adam_m_(k, g) = settings_.adam_beta1 * adam_m_(k, g) +
            //                 (1.0 - settings_.adam_beta1) * gradients_matrix_(k, g);
            // adam_v_(k, g) = settings_.adam_beta2 * adam_v_(k, g) +
            //                 (1.0 - settings_.adam_beta2) * std::pow(gradients_matrix_(k, g), 2);
            // adam_m_(k, g) /= (1.0 - std::pow(settings_.adam_beta1, adam_t_));
            // // adam_v_(k, g) /= (1.0 - std::pow(settings_.adam_beta2, adam_t_));
            // fp_type m_hat = adam_m_(k, g) / (1.0 - std::pow(settings_.adam_beta1, adam_t_));
            // fp_type v_hat = adam_v_(k, g) / (1.0 - std::pow(settings_.adam_beta2, adam_t_));
            // weights_(k, g) -= learning_rate_ * m_hat
            //                 / (std::sqrt(v_hat) + settings_.adam_epsilon);

            // delta_weights_(k, g) = delta_weights_(k, g) * settings_.momentum +
            //                 gradients_matrix_(k, g) * (1.0 - settings_.momentum);
                        
            // adam_cache_(k, g) = settings_.adam_beta * adam_cache_(k, g) +
            //                 (1.0 - settings_.adam_beta) * std::pow(gradients_matrix_(k, g), 2);
            // weights_(k, g) += learning_rate_ * delta_weights_(k, g)
            //                 / (std::sqrt(adam_cache_(k, g)) + settings_.adam_epsilon);
        }
    }
    ++adam_t_;
    // adam_cache_ = weights_ + delta_weights_;

    std::fill(gradients_sum_.begin(), gradients_sum_.end(), 0.0);
    std::fill(gradients_matrix_.DataVector().begin(), gradients_matrix_.DataVector().end(), 0.0);
}

void MatrixLayer::UpdateWeights(int threads)
{
    for (unsigned g = 0; g < destination_.size(); ++g)
    {
        biases_[g] += learning_rate_ * gradients_sum_[g];
    }

    // for (unsigned k = 0; k < weights_.GetRows(); ++k)
    // {
    //     for (unsigned g = 0; g < weights_.GetCols(); ++g)
    //     {
    //         delta_weights_(k, g) = delta_weights_(k, g) * settings_.momentum +
    //                         gradients_matrix_(k, g) * (1.0 - settings_.momentum);
    //         weights_(k, g) += delta_weights_(k, g);
    //     }
    // }
    std::vector<std::thread> tv;
    for (unsigned thc = 0; thc < threads; ++thc)
    {
        tv.emplace_back([&, thc] {
            unsigned delta = weights_.GetRows() / threads;
            unsigned end = (thc == threads - 1 ? weights_.GetRows() : (thc + 1) * delta);
            for (unsigned k = thc * delta; k < end; ++k)
            {
                for (unsigned g = 0; g < weights_.GetCols(); ++g)
                {
                    delta_weights_(k, g) = delta_weights_(k, g) * settings_.momentum +
                                    gradients_matrix_(k, g) * (1.0 - settings_.momentum);
                    weights_(k, g) += delta_weights_(k, g);
                }
            }
        });
    }
    for (auto &thread : tv)
    {
        thread.join();
    }

    // for (unsigned thc = 0; thc < threads; ++thc) {
    //     thrs.emplace_back([&, thc](){
    //         unsigned delta = n / threads;
    //         unsigned end = (thc == threads - 1 ? n : (thc + 1) * delta);
    //         for (unsigned i = thc * delta; i < end; ++i) {
    //             for (unsigned j = 0; j < n; ++j) {
    //                 T sum = 0;
    //                 for (unsigned l = 0; l < n; ++l) {
    //                     sum += A[i * n + l] * B[l * n + j];
    //                 }
    //                 C[i * n + j] = sum;
    //             }
    //         }
    //     });
    // }

    std::fill(gradients_sum_.begin(), gradients_sum_.end(), 0.0);
    std::fill(gradients_matrix_.DataVector().begin(), gradients_matrix_.DataVector().end(), 0.0);
}


void MatrixModel::UpdateWeights()
{   
    for (auto &layer : layers_)
    {
        layer.UpdateWeights();
    }
}

//////////////////////////////////////////////////////////////////////////
// OTHER

int MatrixModel::GetResult()
{
    fp_type max = -std::numeric_limits<fp_type>::infinity();
    int result = 0;
    const auto &output = layers_.back().destination_;
    for (unsigned k = 0; k < output.size(); ++k) 
    {
        if (max < output[k])
        {
            max = output[k];
            result = k;
        }
    }
    return result;
}

void MatrixModel::UpdateLR()
{
    for (auto &i : layers_)
    {
        i.learning_rate_ *= settings_.lr_epoch_k;
    }
}

//////////////////////////////////////////////////////////////////////////
// FILE


MatrixLayer::MatrixLayer(size_t rows, size_t cols, fp_type lr, const PerceptronSettings &settings, std::ifstream &file)
    : weights_(rows, cols, 0)
    , delta_weights_(rows, cols, 0)
    , biases_(cols)
    , destination_(cols)
    , gradients_(cols)
    , settings_(settings)
    , learning_rate_(lr)
{    
    std::string line;
    std::getline(file, line);
    if (line != "weights:")
    {
        file.close();
        std::cout << cols << ' ' << rows << '\n';
        throw std::runtime_error("MatrixLayer: Invalid file format in line \"weight:\": " + line);
    }
    for (unsigned int i = 0; i < weights_.GetRows(); ++i)
    {
        for (unsigned int j = 0; j < weights_.GetCols(); ++j)
        {
            file >> weights_(i, j);
        }
    }
    file .ignore(2);
    std::getline(file, line);
    if (line == "biases:")
    {
        for (auto &b : biases_)
        {
            file >> b;
        }
        file .ignore(2);
    }
    else if (line != "empty biases")
    {
        file.close();
        throw std::runtime_error("MatrixLayer: Invalid file format in line \"biases:\": " + line);
    }

}


MatrixModel::MatrixModel(const std::string &file_name)
{
    std::ifstream file(file_name);
    settings_ = PerceptronSettings(file);

    fp_type lr = settings_.learning_rate;
    for (unsigned int k = 0; k < settings_.layers.size() - 1; ++k)
    {
        layers_.emplace_back(settings_.layers[k], settings_.layers[k + 1], lr, settings_, file);
        lr *= settings_.lr_layers_k;
    }
    
    file.close();

}

void MatrixModel::ToFile(const std::string &file_name)
{
    std::ofstream file(file_name);
    settings_.ToFile(file);

    for (auto &layer : layers_)
    {
        file << "weights:\n";
        auto &lw = layer.weights_;
        for (unsigned int i = 0; i < lw.GetRows(); ++i)
        {
            for (unsigned int j = 0; j < lw.GetCols(); ++j)
            {
                file << lw(i, j) << ' ';
            }
        }
        file << "\nbiases:\n";
        for (auto &b : layer.biases_)
        {
            file << b << ' ';
        }
        file << '\n';
    }
    file.close();
}

