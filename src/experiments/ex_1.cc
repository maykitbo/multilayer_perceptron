// #include "../model/data_manager.h"
#include "../model/matrix_model.h"
// #include "../lib/m_time.h"
// #include <thread>

using namespace s21;

int main(int argc, char *argv[])
{


    std::vector<int> S{784, 96, 64, 48, 32, 29, 26};

    size_t M = 0;

    for (int k = 0; k < S.size() - 1; ++k)
    {
        M += S[k] * S[k + 1] * 3 + S[k + 1] * 4;
    }
    M *= sizeof(double);
    std::cout << M << '\n';
    std::cout << sizeof(double) << '\n';
    return 0;


    std::string file_path = "../datasets/emnist-letters-train.csv";
    int bias = -1;
    size_t width = Const::letter_width;
    size_t height = Const::letter_height;
    unsigned classes = Const::min_layer;
    DataManager letters(file_path, bias, kNoRotate, width, height, classes);
    // letters.Split(10);

    std::cout << "Dataset size: " << letters.Size() << "\n";

    PerceptronSettings settings;
    settings.layers = {Const::max_layer, 200, 150, 100, Const::min_layer};
    settings.learning_rate = 0.001;
    settings.lr_epoch_k = 0.98;
    settings.lr_layers_k = 0.99;
    settings.momentum = 0.2;

    settings.SetActivation(kReLU);
    settings.SetWeightInit(kHE);

    // settings.SetActivation(kReLU);
    // settings.SetWeightInit(kHE);

    MatrixModel model(settings);

    bool stop = false;
    fp_type prev_error = 1.0;

    std::vector<double> errors;

    model.SetErrorThread([&stop, &prev_error] (fp_type error, unsigned epoch) {
        std::cout << "Epoch: " << epoch << " Error: " << error << "\n";
    });
    model.SetMetricThread([] (const Metrics &m) {
        std::cout << "Accutacy: " << m.accuracy << "   ";
        // for (auto p : m.precision) std::cout << p << ' ';
        std::cout << '\n';
    });


    unsigned epoch_count = 3;
    unsigned batch = 64;
    std::cout << Time::Test([&] {
        model.Train(letters, epoch_count, batch, stop);
    }, 1) << '\n';

    return 0;
}