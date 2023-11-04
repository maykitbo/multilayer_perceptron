#include <QImage>
#include "../model/data_manager.h"
#include "../model/matrix_model.h"
#include "../lib/m_time.h"
#include "metrics_widget/metrics_widget.h"
#include <QApplication>

using namespace s21;

void Rep1Helper(int name, int d, QImage &image)
{
    QTransform transform;
    transform.rotate(d);
    QImage rotatedImage = image.transformed(transform, Qt::FastTransformation);

    int newSize = 28 * 5;
    QImage finalImage(newSize, newSize, QImage::Format_RGBA8888);
    finalImage.fill(Qt::transparent);

    // Calculate the offset to center the rotated image
    int offsetX = (newSize - rotatedImage.width()) / 2;
    int offsetY = (newSize - rotatedImage.height()) / 2;

    // Copy the rotated image to the center of the final image
    QPainter painter(&finalImage);
    painter.drawImage(offsetX, offsetY, rotatedImage);

    painter.setPen(QPen(MetricsWidgetStyle::axes_color, 1));
    painter.drawRect(0, 0, finalImage.width() - 1, finalImage.height() - 1);

    finalImage.save("../misc/LettersRep/ex1_" + QString(QChar(name + 'a')) + "_" + QString::number(d) + ".png");
}
void applyBlur(int name, QImage &image, int radius)
{
    QImage blurredImage = image;
    for (int y = 0; y < image.height(); ++y)
    {
        for (int x = 0; x < image.width(); ++x)
        {
            int redSum = 0, greenSum = 0, blueSum = 0, alphaSum = 0, pixelCount = 0;
            for (int dy = -radius; dy <= radius; ++dy)
            {
                for (int dx = -radius; dx <= radius; ++dx)
                {
                    int newX = qBound(0, x + dx, image.width() - 1);
                    int newY = qBound(0, y + dy, image.height() - 1);
                    QColor pixelColor = image.pixelColor(newX, newY);
                    redSum += pixelColor.red();
                    greenSum += pixelColor.green();
                    blueSum += pixelColor.blue();
                    alphaSum += pixelColor.alpha();
                    ++pixelCount;
                }
            }
            int avgRed = redSum / pixelCount;
            int avgGreen = greenSum / pixelCount;
            int avgBlue = blueSum / pixelCount;
            int avgAlpha = alphaSum / pixelCount;
            blurredImage.setPixelColor(x, y, QColor(avgRed, avgGreen, avgBlue, avgAlpha));
        }
    }
    blurredImage = blurredImage.scaled(28 * 5, 28 * 5);

    QPainter painter(&blurredImage);
    painter.setPen(QPen(MetricsWidgetStyle::axes_color, 1));
    painter.drawRect(0, 0, blurredImage.width() - 1, blurredImage.height() - 1);
    blurredImage.save("../misc/LettersRep/ex2_" + QString(((char)name + 'a')) + "_" + QString::number(radius) + ".png");
}

void Rep1(dataset_t &data)
{
    for (int k = 0; k < 4; ++k)
    {
        unsigned name = data[k].second;
        data_vector letter(data[k].first);
        QImage image(28, 28, QImage::Format_RGBA8888);
        image.fill(Qt::transparent);
        for (unsigned i = 0; i < 28; ++i)
        {
            for (unsigned j = 0; j < 28; ++j)
            {
                image.setPixelColor(i, j, QColor(255, 0, 0, letter[i * 28 + j] * 255));
            }
        }
        applyBlur(name, image, 1);
        applyBlur(name, image, 2);
        // applyBlur(name, image, 5);
        image = image.scaled(28 * 5, 28 * 5);
        Rep1Helper(name, 0, image);
        Rep1Helper(name, 15, image);
        Rep1Helper(name, -15, image);
        
        
    }
}

// using namespace std;


struct Settings
{
    vector<size_t> layers;         ///<  Размеры слоёв
    float learning_rate;           ///<  Скорость обучения (0.001, 0.002)
    float momentum = 0.0;          ///<  Коэффициент инерции (0.0, 0.9)
    float lr_epoch_k = 1.0;        ///<  Изменение lr с каждой эпохой
    float lr_layers_k = 1.0;       ///<  Изменение lr по слоям
    float weight_mean = 0.0;       ///<  Среднее значение весов
    float weight_sd = 0.01;        ///<  Среднее отклонение весов
    /* ... */

    function<float(float)> activation;                     ///<  Функция активации
    function<float(float)> derivative_activation;          ///<  Производная активации
    function<float(int, int)> weight_init;                 ///<  Функция инициализации весов
    function<vector<float>(vector<float>)> loss_function;  ///<  Функция потерь
};
// using vector = std::vector;
// using matrix = std::pair<int, int>;

template<class T>
struct vector {
    vector(T size) {}
};

template<class T>
struct matrix {
    matrix(T cols, T rows) {}
};

struct Layer
{
    int output;
    int biases;
    int weights;

    int gradients_matrix;
    int gradients_sum;
    int gradients;
    int learning_rate;

    // Layer(int cols, int rows);

    void CollectGradients(vector &a, vector &b);
    void ProcessTarget(int target);
    void ProcessGradient();
    void PassGradient(Layer &prev_layer);
    void UpdateWeights();
};

// Layer::Layer(int cols, int rows)
//     : output(cols)
//     , biases(cols)
//     , weights(cols, rows)
//     , gradients_matrix(cols, rows)
//     , gradients_sum(cols)
//     , gradients(cols)
//     void OutputBackward(Layer &prev_layer, int target);
//     void InputBackward(vector<float> &input);
//     void HiddenBackward(Layer &prev_layer);
// {}

struct Settings
{
    float learning_rate;
    float Activation(float x);
    float DerivativeActivation(float x);
    /* ... */
};

struct NeronNetwork
{
    vector<Layer> layers;
    Settings settings;
    void Backward(vector &input, int target);
    void UpdateWeights();
};

void NeronNetwork::UpdateWeights()
{
    for (auto &layer : layers)
    {
        layer.UpdateWeights();
    }
}

void Layer::UpdateWeights()
{
    biases += learning_rate * gradients_sum;
    weights += learning_rate * gradients_matrix;

    fill(gradients_sum.begin(), gradients_sum.end(), 0.0);
    gradients_matrix.fill(0.0);
}

int MulABT(int a, int b)
{
    return 0;
}

void Layer::CollectGradients(vector &a, vector &b)
{
    // matrix = matrix + vector^T * vector
    gradients_matrix += gradients * prev_layer.output;
    gradients_sum += gradients;
}

void Layer::ProcessTarget(int target)
{
    gradients = -output;
    gradients[target] += 1.0;
}

void Layer::ProcessGradient()
{
    gradients *= derivative_activation(output);
}

void Layer::PassGradient(Layer &prev_layer)
{
    // vector = vector * matrix^T
    prev_layer.gradients = MulABT(gradients, weights);
}


void NeronNetwork::Backward(vector &input, int target)
{
    int k = layers.size() - 1;
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


void NeronNetwork::Forward(vector &input)
{
    // для каждого слоя кроме последнего
    for (int i = 0; i < layers.size() - 1; ++i)
    {
        // input первого слоя - входные данные
        layers[i].HiddenSignal(input);
        // output слоя - input следующего слоя
        input = layers[i].output;
    }
    // последний слой
    layers.back().OutputSignal(input);
}

void Layer::HiddenSignal(vector &input)
{
    // vector = vector * matrix
    output = input * weights;
    // vector = f(vector + vector)
    output = activation(output + biases);
}

void Layer::OutputSignal(vector &input)
{
    // vector = vector * matrix
    output = input * weights;
    // vector = f(vector + vector)
    output = lossfunc(output + biases);
}

// matrix_t weights_, delta_weights_;
//         matrix_t gradients_matrix_;
//         std::vector<fp_type> biases_, destination_, error_;
//         std::vector<fp_type> gradients_, gradients_sum_;
//         const PerceptronSettings &settings_;
//         fp_type learning_rate_;

int main(int argc, char *argv[])
{

    DataManager letters("../datasets/emnist-letters-train.csv", -1, kNoRotate);
    dataset_t &data = letters.GetLetters();
    std::vector<double> letters_count(26, 0.0);
    for (auto &d : data)
    {
        letters_count[d.second] += Random::Uniform(3.0, 6.0);
        // letters_count[d.second] += 1.0;
    }
    Rep1(data);
    for (auto &lc : letters_count)
    {
        lc += Random::Uniform(-4000.0, 4000.0);
    }

    QApplication a(argc, argv);
    MetricsWidget w;
    w.show();
    w.SetVector(letters_count);
    return a.exec();



    // for (int k = 0; k < 10; ++k)
    // {
    //     unsigned name = data[k].second;
    //     data_vector letter(data[k].first);
    //     QImage image(28, 28, QImage::Format_RGBA8888);
    //     image.fill(Qt::transparent);
    //     for (unsigned i = 0; i < 28; ++i)
    //     {
    //         for (unsigned j = 0; j < 28; ++j)
    //         {
    //             image.setPixelColor(i, j, QColor(255, 0, 0, letter[i * 28 + j] * 255));
    //         }
    //     }
    
    //     image = image.scaled(28 * 5, 28 * 5);
    //     QPainter painter(&image);
    //     painter.setPen(QPen(MetricsWidgetStyle::axes_color, 1));
    //     painter.drawRect(0, 0, image.width() - 1, image.height() - 1);
    //     image.save("../misc/LettersEx/good_" + QString(((char)name + 'a')) + QString::number(letters_count[name]) + ".png");
    // }
    // return 0;

    // QImage image(140, 140, QImage::Format_RGBA8888);
    // {
    //     QPainter painter(&image);
    //     painter.setPen(QPen(MetricsWidgetStyle::axes_color, 1));
    //     painter.drawRect(0, 0, image.width() - 1, image.height() - 1);


    //     QFont font;
    //     font.setPointSize(100); // Set the font size to make the letter A larger
    //     painter.setFont(font);

    //     painter.setPen(QPen(Qt::red, 100));
    //     QRect rect(0, 0, image.width(), image.height());
    //     painter.drawText(rect, Qt::AlignCenter, "A");
    // }
    // image.save("../misc/LettersEx/bad_A_5.png");
    // return a.exec();

    for (int k = 0; k < 5; ++k) {
    unsigned name = data[k].second;
    data_vector letter(data[k].first);
    QImage image(28, 28, QImage::Format_RGBA8888);
    image.fill(Qt::transparent);
    for (unsigned i = 0; i < 28; ++i)
    {
        for (unsigned j = 0; j < 28; ++j)
        {
            image.setPixelColor(i, j, QColor(255, 0, 0, letter[i * 28 + j] * 255));
        }
    }

    for (int h = 0; h < 30; ++h) {
        image.setPixelColor(Random::Int(0, 8), Random::Int(0, 8), QColor(255, 0, 0, Random::Int(0, 100)));
    }

    image = image.scaled(28 * 5, 28 * 5);

    

    // QTransform transform;
    // transform.rotate(45);
    // image = image.transformed(transform);

    // image = image.scaled(28 * 5, 28 * 5);

    QPainter painter(&image);
    painter.setPen(QPen(MetricsWidgetStyle::axes_color, 1));
    painter.drawRect(0, 0, image.width() - 1, image.height() - 1);
    image.save("../misc/LettersEx/bad_" + QString(((char)name + 'a')) + QString::number(letters_count[name]) + "_6.png");
    }

    return 0;
}