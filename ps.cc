#include <vector>
#include <functional>
using namespace std;

template<class T>
struct Matrix
{
    Matrix(size_t rows, size_t cols) : rows(rows) {}
};


typedef vector<float> vector_t;
typedef Matrix<float> matrix_t;

struct Layer
{
    vector_t output;
    vector_t biases;
    matrix_t weights;

    vector_t gradient;
    vector_t gradient_sum;
    matrix_t gradient_matrix;

    Layer(size_t rows, size_t cols);

    void HiddenSignal(vector_t *input);
    void OutputSignal(vector_t *input);
    void ProcessTarget(uint target);
    void ProcessGradient();
    void PassGradient(Layer &prev_layer);
    void CollectGradients();
    void UpdateWeights();
};

Layer::Layer(size_t rows, size_t cols)
    : output(cols)
    , biases(cols)
    , weights(rows, cols)

    , gradient(cols)
    , gradient_sum(cols)
    , gradient_matrix(rows, cols)
{}

struct Settings
{
    vector<size_t> layers;
    float learning_rate;
    function<float(float)> activation;
    function<float(float)> derivative_activation;
};

struct NeuronNetwork
{
    vector<Layer> layers;
    Settings settings;
    void Forward(vector_t &letter);
    void UpdateWeights();
};

void Layer::CollectGradients()
{
    // matrix = matrix + vector^T * vector
    gradient_matrix += gradient * prev_layer.output;
    gradient_sum += gradient;
}

void Layer::ProcessTarget(uint target)
{
    gradient = -output;
    gradient[target] += 1.0;
}

void Layer::ProcessGradient()
{
    gradient *= DerivativeActivation(output);
}

void Layer::PassGradient(Layer &prev_layer)
{
    // vector = vector * matrix^T
    prev_layer.gradient = MulABT(gradient, weights);
}


void NeuronNetwork::Backward(vector_t &input
                            , uint target)
{
    size_t k = layers.size() - 1;
    // последний слой
    layers[k].ProcessTarget(target);
    layers[k].PassGradient(layers[k - 1]);
    layers[k].CollectGradients();

    // для каждого скрытого слоя
    for (--k; k >= 1; --k)
    {
        layers[k].ProcessGradient();
        layers[k].PassGradient(layers[k - 1]);
        layers[k].CollectGradients();
    }

    // первый слой
    layers[0].ProcessGradient();
    layers[0].CollectGradients();
}

int learning_rate;

// после каждого батча:
void NeuronNetwork::UpdateWeights()
{
    for (auto &layer : layers)
    {
        layer.UpdateWeights();
    }
}

void Layer::UpdateWeights()
{
    biases += learning_rate * gradient_sum;
    weights += learning_rate * gradient_matrix;

    // обнуляем градиенты
    fill(gradients_sum.begin(), gradients_sum.end(), 0.0);
    gradients_matrix.fill(0.0);
}



void NeuronNetwork::Forward(vector_t &letter)
{
    // вход первого слоя - входной вектор
    vector_t input = &letter;
    // для каждого слоя кроме последнего
    for (size_t i = 0; i < layers.size() - 1; ++i)
    {
        layers[i].HiddenSignal(input);
        // выход текущего слоя - вход следующего
        letter = &layers[i].output;
    }
    // последний слой
    layers.back().OutputSignal(input);
}

void Layer::HiddenSignal(vector_t *input)
{
    // vector = vector * matrix
    output = (*input) * weights;
    // vector = f(vector + vector)
    output = Actvation(output + biases);
}

void Layer::OutputSignal(vector_t *input)
{
    // vector = vector * matrix
    output = (*input) * weights;
    // vector = f(vector + vector)
    output = LossFunction(output + biases);
}
