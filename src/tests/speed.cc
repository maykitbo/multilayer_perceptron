#include "test.h"

#include "../lib/m_time.h"
#include "../lib/random.h"

using namespace s21;

void s21Test(DataManager& data) {
    MatrixModel MM(rp("materials/perceptron_matrix.prcp"));
    GraphModel GM(rp("materials/perceptron_matrix.prcp"));
    data.Split(15);
    for (auto iter : {10, 100, 1000}) {
        auto result = Time::Compare(iter, [&] {
            MM.Test(data);
        }, [&] {
            GM.Test(data);
        });
        std::cout << "\033[1;32m" << iter << " iterations:\033[0m\n";
        std::cout << "\tMatrix: " << result[0] << "\n";
        std::cout << "\tGraph: " << result[1] << "\n";
    }
}

void Test(PerceptronSettings &ps, unsigned int repeat = 10) {
    static int counter = 0;
    std::vector<fp_type> letter(784);
    int ans = Random::Int(0, 26);
    for (int i = 0; i < 784; ++i) {
        letter[i] = Random::Normal<fp_type>(0.0, 1.0);
    }

    MatrixModel MM(ps);
    MM.SetLetter(&letter);
    GraphModel GM(ps);
    GM.SetLetter(&letter);

    auto forward_resalt = Time::Compare<Time::ns>(repeat, [&] {
        MM.Forward();
    }, [&] {
        GM.Forward();
    });
    auto backward_result = Time::Compare<Time::ns>(repeat, [&] {
        MM.Forward();
        MM.Backward(ans);
    }, [&] {
        GM.Forward();
        GM.Backward(ans);
    });
    std::cout << "\033[1;31m" << "Test #" << ++counter << ":\033[0m" << " {";
    for (unsigned int l = 0; l < ps.layers.size() - 1; ++l) {
        std::cout << ps.layers[l] << ", ";
    }
    std::cout << ps.layers.back() << "}\n";
    std::cout << "\033[1;32m" << "Forward:\033[0m\n";
    std::cout << "\tMatrix: " << forward_resalt[0] << " ns\n";
    std::cout << "\tGraph: " << forward_resalt[1] << " ns\n";
    std::cout << "\033[1;32m" << "Forward + Backward:\033[0m\n";
    std::cout << "\tMatrix: " << backward_result[0] << " ns\n";
    std::cout << "\tGraph: " << backward_result[1] << " ns\n\n";
};

int main() {

    std::cout << "Whould you like to run first test (required for school task)? (y/n)\n";
    char c;
    std::cin >> c;
    if (c == 'y') {
        DataManager data(rp("datasets/emnist-letters-train.csv"), -1, k90Rotate);
        std::cout << "\033[1;31m" << "s21Test:" << "\033[0m\n";
        s21Test(data);
        std::cout << "/n/n";
    }

    std::cout << "Whould you like to run next test (not required for school task)? (y/n)\n";
    std::cin >> c;
    if (c != 'y') {
        return 0;
    }

    PerceptronSettings ps = CorrectSettings();

    std::vector<size_vector> layers = {{784, 26},
                                        {784, 32, 26},
                                        {784, 64, 32, 26},
                                        {784, 128, 64, 32, 26},
                                        {784, 256, 128, 64, 32, 26},
                                        {784, 256, 256, 26}};
    for (const auto &layers : layers) {
        ps.layers = layers;
        Test(ps, 10000);
    }
    return 0;
}
