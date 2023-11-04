#include "../model/data_manager.h"
#include "../model/matrix_model.h"
#include "../lib/m_time.h"
// #include "metrics_widget/metrics_widget.h"
// #include "../image_painter/image_painter.h"
// #include <QApplication>
#include <fstream>
#include <thread>

using namespace s21;

void OneTest1(int r, PerceptronSettings &settings)
{
    static int i = 4;
    std::string file_path = "../datasets/emnist-letters-train.csv";
    int bias = -1;
    size_t width = Const::letter_width;
    size_t height = Const::letter_height;
    unsigned classes = Const::min_layer;
    DataManager letters(file_path, bias, kNoRotate, width, height, classes);
    // letters.Split(10);

    if (r != 0)
    {
        // letters.IncreaseRoatation(-r);
        letters.IncreaseRoatation(r);
    }
    
    // return 0;

    std::cout << "Dataset size: " << letters.Size() << "\n";

    

    // settings.SetActivation(kReLU);
    // settings.SetWeightInit(kHE);

    MatrixModel model(settings);

    bool stop = false;
    fp_type prev_error = 1.0;

    std::vector<std::pair<unsigned, double>> errors;
    std::vector<double> accurs;
    std::vector<std::vector<double>> precisions;
    std::vector<std::vector<double>> recalls;
    std::vector<std::vector<double>> f1s;

    model.SetErrorThread([&stop, &prev_error, &errors] (fp_type error, unsigned epoch) {
        // std::cout << "Epoch: " << epoch << " Error: " << error << "\n";
        errors.emplace_back(epoch, error);
    });
    model.SetMetricThread([&accurs, &precisions, &recalls, &f1s] (const Metrics &m) {
        // std::cout << "Accutacy: " << m.accuracy << "\n";
        // for (auto p : m.precision) std::cout << p << ' ';
        // std::cout << '\n';
        accurs.push_back(m.accuracy);
        precisions.push_back(m.precision);
        recalls.push_back(m.recall);
        f1s.push_back(m.F1);
    });


    unsigned epoch_count = 6;
    unsigned batch = 64;
    auto time = Time::Test([&] {
        model.Train(letters, epoch_count, batch, stop);
    }, 1);

    std::cout << time << "   " << accurs.back() << "\n\n";

    std::ofstream file("test_3_" + std::to_string(i) + "_" + std::to_string(r) + ".txt");
    settings.ToFile(file);
    file << "Time = " << time << '\n';
    file << epoch_count << ' ' << batch << '\n';
    for (auto e : errors)
        file << e.first << ' ' << e.second << '\n';
    file << "\n\n";
    for (auto a : accurs)
        file << a << '\n';
    
    for (auto p : precisions)
    {
        for (auto v : p)
            file << v << ' ';
        file << '\n';
    }
    file << "\n\n";
    for (auto r : recalls)
    {
        for (auto v : r)
            file << v << ' ';
        file << '\n';
    }
    file << "\n\n";
    for (auto f : f1s)
    {
        for (auto v : f)
            file << v << ' ';
        file << '\n';
    }

    file.close();
    ++i;
}

int main(int argc, char *argv[])
{

    // QApplication a(argc, argv);

    // Graph::ImagePainter painter(QSize(1600, 900), argc, argv);
    // painter.ReserveGraphs(1);

    PerceptronSettings settings;
    settings.layers = {Const::max_layer, 200, 150, 100, Const::min_layer};
    settings.learning_rate = 0.001;
    settings.lr_epoch_k = 0.98;
    settings.lr_layers_k = 0.97;
    settings.momentum = 0.5;
    settings.SetActivation(kReLU);
    settings.SetWeightInit(kHE);
    OneTest1(0, settings);
    OneTest1(10, settings);
    OneTest1(14, settings);

    // settings.learning_rate = 0.0008;
    // settings.lr_epoch_k = 1.4;
    // settings.lr_layers_k = 2.2;
    // settings.momentum = 0.8;

    // settings.SetActivation(kTanh);
    // settings.SetWeightInit(kXavier);
    // OneTest1(0, settings);
    // OneTest1(10, settings);
    // OneTest1(14, settings);

    // settings.SetActivation(kSiLU);
    // settings.SetWeightInit(kXavier);
    // OneTest1(0, settings);
    // OneTest1(10, settings);
    // OneTest1(14, settings);



    // settings.layers = {Const::max_layer, 300, 300, Const::min_layer};
    // settings.SetActivation(kSigmoid);
    // settings.SetWeightInit(kHE);
    // OneTest1(0, settings);

    // settings.SetActivation(kReLU);
    // settings.SetWeightInit(kHE);
    // OneTest1(0, settings);

    // settings.SetActivation(kTanh);
    // settings.SetWeightInit(kXavier);
    // OneTest1(0, settings);

    // settings.SetActivation(kSiLU);
    // settings.SetWeightInit(kXavier);
    // OneTest1(0, settings);

    // settings.layers = {Const::max_layer, 200, 150, 100, Const::min_layer};
    // settings.momentum = 0.4;
    // settings.SetActivation(kReLU);
    // settings.SetWeightInit(kHE);
    // OneTest1(0, settings);

    // settings.SetActivation(kTanh);
    // settings.SetWeightInit(kXavier);
    // OneTest1(0, settings);

    // settings.SetActivation(kSiLU);
    // settings.SetWeightInit(kXavier);
    // OneTest1(0, settings);

    // settings.layers = {Const::max_layer, 300, Const::min_layer};
    // settings.SetActivation(kSigmoid);
    // settings.SetWeightInit(kHE);
    // OneTest1(0, settings);

    // settings.SetActivation(kReLU);
    // settings.SetWeightInit(kHE);
    // OneTest1(0, settings);

    // settings.SetActivation(kTanh);
    // settings.SetWeightInit(kXavier);
    // OneTest1(0, settings);

    // settings.SetActivation(kSiLU);
    // settings.SetWeightInit(kXavier);
    // OneTest1(0, settings);


    // settings.layers = {Const::max_layer, 1024, 256, 64, Const::min_layer};
    // settings.SetActivation(kSigmoid);
    // settings.SetWeightInit(kHE);
    // OneTest1(0, settings);

    // settings.SetActivation(kReLU);
    // settings.SetWeightInit(kHE);
    // OneTest1(0, settings);

    // settings.SetActivation(kTanh);
    // settings.SetWeightInit(kXavier);
    // OneTest1(0, settings);

    // settings.SetActivation(kSiLU);
    // settings.SetWeightInit(kXavier);
    // OneTest1(0, settings);
    


    return 0;
    // return a.exec();
}