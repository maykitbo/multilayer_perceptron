// #include "../model/data_manager.h"
// #include "../model/matrix_model.h"
#include "../../lib/m_time.h"
#include "../metrics_widget/metrics_widget.h"
#include "image_painter/image_painter.h"
#include <QApplication>
#include <fstream>
#include <locale.h>
#include "../../common/metrics.h"
#include <filesystem>
#include <sstream>



Graph::data_t Parser(const std::string &file_name)
{
    Graph::data_t data;
    std::ifstream file(file_name);
    double x_step = 1.0 / 9.0;

    for (int k = 0; k < 13; ++k)
    {
        file.ignore(1000, '\n');
    }
    double x = 0.0;
    int t = 0;
    for (int k = 0; k < 54; ++k)
    {
        double y;
        file >> t;
        file >> y;
        data.push_back({x, y});
        x += x_step;
    }
    file.close();
    return data;
}

std::vector<s21::Metrics> MetricsParser(const std::string &file_name)
{
    std::vector<s21::Metrics> result(6, s21::Metrics(26));
    std::ifstream file(file_name);
    for (int k = 0; k < 13 + 54 + 2; ++k)
    {
        file.ignore(1000, '\n');
    }
    for (int k = 0; k < 6; ++k)
    {
        double a;
        file >> a;
        result[k].accuracy = a;
    }
    for (int k = 0; k < 6; ++k)
    {
        for (int g = 0; g < 26; ++g)
        {
            double a;
            file >> a;
            result[k].precision[g] = a;
        }
    }
    file.ignore(1000, '\n');
    file.ignore(1000, '\n');
    for (int k = 0; k < 6; ++k)
    {
        for (int g = 0; g < 26; ++g)
        {
            double a;
            file >> a;
            result[k].recall[g] = a;
        }
    }
    file.ignore(1000, '\n');
    file.ignore(1000, '\n');
    for (int k = 0; k < 6; ++k)
    {
        for (int g = 0; g < 26; ++g)
        {
            double a;
            file >> a;
            result[k].F1[g] = a;
        }
    }
    file.close();
    return result;
}

void OnePng(std::string name, int argc, char *argv[])
{
    auto data = Parser(name);
    Graph::ImagePainter painter(QSize(1600, 900), argc, argv);
    painter.ReserveGraphs(1);
    painter.AddGraph(data, " ");
    painter.GetBackground().SetXTextFunc([] (double x) {
        return QString::number(x, 'f', 2);
    });
    painter.GetBackground().SetVerticalGridStep(0.5);
    name.resize(name.size() - 4);
    size_t last_slash = name.find_last_of('/');
    if (last_slash != std::string::npos) {
        name = name.substr(last_slash + 1);
    }
    
    painter.SavePng("img/" + QString::fromStdString(name));
}

#include <cmath>

std::pair<double, double> MeanSD(const std::vector<double> &data)
{
    double sum = 0.0;
    for (auto d : data)
    {
        sum += d;
    }
    double mean = sum / data.size();
    double variance = 0.0;
    for (auto d : data)
    {
        variance += std::pow(d - mean, 2);
    }
    double sd = std::sqrt(variance / data.size());
    return {mean, sd};
}

void OneMetrics(std::string name)
{
    auto data = MetricsParser(name);
    name.resize(name.size() - 4);
    size_t last_slash = name.find_last_of('/');
    if (last_slash != std::string::npos) {
        name = name.substr(last_slash + 1);
    }
    
    std::cout << name << "\n\t";
    std::cout << "Accuracy: " << data.back().accuracy << "\n\t";
    auto p = MeanSD(data.back().precision);
    std::cout << "Precision: " << p.first << " +- " << p.second << "\n\t";
    p = MeanSD(data.back().recall);
    std::cout << "Recall: " << p.first << " +- " << p.second << "\n\t";
    p = MeanSD(data.back().F1);
    std::cout << "F1: " << p.first << " +- " << p.second << "\n\n";
    
}

int main(int argc, char *argv[])
{
    setenv("LC_NUMERIC", "C", 1);
    std::string name = "test_3_0_0";

    OnePng("experiments/graph/data/test_3_0_0.txt", argc, argv);
    OnePng("experiments/graph/data/test_3_1_0.txt", argc, argv);

    OneMetrics("experiments/graph/data/test_3_0_0.txt");
    OneMetrics("experiments/graph/data/test_3_1_0.txt");


    // auto data = MetricsParser("experiments/graph/data/" + name + ".txt");
    // QApplication a(argc, argv);
    // s21::MetricsWidget w;
    // w.show();
    // for (auto &m : data)
    // {
    //     w.SetMetrics(m);
    // }
    // return a.exec();


    // std::string directory_path = "experiments/graph/data/";

    // for (const auto &entry : std::filesystem::directory_iterator(directory_path))
    // {
    //     if (entry.is_regular_file())
    //     {
    //         const std::string &file_name = entry.path().string();
    //         OnePng(file_name, argc, argv);
    //         // OneMetrics(file_name);
    //     }
    // }




    return 0;
}
