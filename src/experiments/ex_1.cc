#include "../model/data_manager.h"
#include "../model/matrix_model.h"
#include "../lib/m_time.h"

using namespace s21;

int main(int argc, char *argv[])
{
    std::string file_path = "../datasets/emnist-letters-train.csv";
    int bias = -1;
    size_t width = Const::letter_width;
    size_t height = Const::letter_height;
    unsigned classes = Const::min_layer;
    DataManager letters(file_path, bias, kNoRotate, width, height, classes);
    // letters.Split(10);

    PerceptronSettings settings;
    settings.layers = {Const::max_layer, 200, 150, 100, Const::min_layer};
    settings.learning_rate = 0.001;
    settings.lr_epoch_k = 0.98;
    settings.lr_layers_k = 0.99;
    settings.momentum = 0.2;
    settings.SetActivation(kSiLU);
    settings.SetWeightInit(kXavier);

    MatrixModel model(settings);

    bool stop = false;
    fp_type prev_error = 1.0;
    model.SetErrorThread([&stop, &prev_error] (fp_type error, unsigned epoch) {
        std::cout << "Epoch: " << epoch << " Error: " << error << "\n";
        if (prev_error < error)
        {
            stop = true;
            std::cout << "Error increased, stopping...\n";
        }
        prev_error = error;
    });
    model.SetMetricThread([] (const Metrics &m) {
        std::cout << "Accutacy: " << m.accuracy << "\n";
    });


    unsigned epoch_count = 10;
    unsigned batch = 64;
    std::cout << Time::Test([&] {
        model.Train(letters, 10, batch, stop);
    }, 1) << '\n';

    return 0;
}