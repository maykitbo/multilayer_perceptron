#include "matrix_model.h"

using namespace s21;

/////////////////////////////////////////////////////////////////////
// CONSTRUCTORS

MatrixLayer::MatrixLayer(size_t rows, size_t cols, fp_type lr, const PerceptronSettings &settings)
    : weights_(rows, cols, [&] { return settings.weight_init(rows, cols); })
    , delta_weights_(rows, cols, 0)
    , biases_(cols)
    , destination_(cols)
    , gradients_(cols)
    , error_(cols)
    , settings_(settings)
    , learning_rate_(lr)
{}


MatrixModel::MatrixModel(const PerceptronSettings &settings)
{
    settings.Validate();
    settings_ = settings;
    fp_type lr = settings_.learning_rate;
    for (unsigned int k = 0; k < settings_.layers.size() - 1; ++k)
    {
        layers_.emplace_back(settings_.layers[k], settings_.layers[k + 1], lr, settings_);
        lr *= settings_.lr_layers_k;
    }
}

/////////////////////////////////////////////////////////////////////////
// FORWARD

void MatrixLayer::Signal(const std::vector<fp_type> *source)
{
    // destination_ = source * weights_ + biases_
    // matrix_t::Mul(*source, weights_, destination_);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        1, weights_.GetCols(), source->size(), 1.0,
        source->data(), source->size(), weights_.Data(),
        weights_.GetCols(), 0.0, destination_.data(), weights_.GetCols());

    for (unsigned int g = 0; g < weights_.GetCols(); ++g)
    {
        destination_[g] = settings_.activation(destination_[g] + biases_[g]);
    }

}

void MatrixModel::Forward()
{
    auto layer = layers_.begin();

    layer->Signal(letter_);
    auto cur = &layer->destination_;

    for (++layer; layer != layers_.end(); ++layer)
    {
        layer->Signal(cur);
        cur = &layer->destination_;
    }
}

////////////////////////////////////////////////////////////////////////
// BACKWARD

void MatrixLayer::UpdateWeights(const std::vector<fp_type> *source)
{
    for (unsigned int k = 0; k < weights_.GetRows(); ++k)
    {
        for (unsigned int g = 0; g < weights_.GetCols(); ++g)
        {
            delta_weights_(k, g) = delta_weights_(k, g) * settings_.momentum +
                            learning_rate_ * gradients_[g] * (*source)[k] * (1.0 - settings_.momentum);
            weights_(k, g) += delta_weights_(k, g);
        }
    }
}

void MatrixLayer::UpdateError(const std::vector<fp_type> &target)
{
    for (unsigned int g = 0; g < error_.size(); ++g)
    {
        error_[g] = target[g] - destination_[g];
    }
}

void MatrixLayer::UpdateGradientsBiases()
{
    fp_type gradient_sum = 0.0;
    for (unsigned int g = 0; g < destination_.size(); ++g)
    {
        gradients_[g] = settings_.derivative_activation(destination_[g]) * error_[g];
        gradient_sum += std::pow(gradients_[g], 2);
        biases_[g] += learning_rate_ * error_[g];
    }
    fp_type l = std::pow(gradient_, 2) * (fp_type)count_ + gradient_sum / gradients_.size();

    ++count_;
    gradient_ = std::sqrt(l / count_);

    if (count_ > std::numeric_limits<int>::max() - 5)
    {
        std::cout << count_ << " COUNT ERROR\n";
    }
}

void MatrixLayer::UpdateFirst(const std::vector<fp_type> *source)
{
    UpdateGradientsBiases();
    UpdateWeights(source);
}

void MatrixLayer::Update(MatrixLayer &prev_layer)
{
    UpdateGradientsBiases();

    // prev_layer.error_ = gradients_ * weights_ ^ T
    // matrix_t::MulBT(gradients_, weights_, prev_layer.error_);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        1, weights_.GetRows(), weights_.GetCols(), 1.0,
        gradients_.data(), weights_.GetCols(), weights_.Data(),
        weights_.GetCols(), 0.0, prev_layer.error_.data(), weights_.GetRows());

    UpdateWeights(&prev_layer.destination_);
}

void MatrixModel::Backward(int answer)
{
    std::vector<fp_type> target(settings_.layers.back(), Const::target.first);
    target[answer] = Const::target.second;

    int layer_k = layers_.size() - 1;
    layers_[layer_k].UpdateError(target);

    for ( ; layer_k >= 1; --layer_k) {
        layers_[layer_k].Update(layers_[layer_k - 1]);
    }

    layers_[0].UpdateFirst(letter_);
}

//////////////////////////////////////////////////////////////////////////
// OTHER

int MatrixModel::GetResult()
{
    fp_type max = -std::numeric_limits<fp_type>::infinity(), result = 0.0;
    const auto &output = layers_.back().destination_;
    for (unsigned int k = 0; k < output.size(); ++k) 
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
    , error_(cols)
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

