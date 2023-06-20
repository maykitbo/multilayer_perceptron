#include "model/data_manager.h"
#include "model/matrix_model.h"
#include "lib/m_time.h"

// typedef double type;
using namespace s21;

int main() {
    int count = 0;

    int CROSS = 9;

    int layers = 4;
    // vector_s layer_sizes1(layers + 1);
    // for (int k = 0; k <= layers; ++k) {
    //     layer_sizes1[k] = Func::LayerReduction(-3.0, k, layers);
    // }

    // vector_s layer_sizes1{Const::max_layer, 200, 150, 100, 50, Const::min_layer};
    // vector_s layer_sizes1{Const::max_layer, 500, 500, Const::min_layer};
    // vector_s layer_sizes1{Const::max_layer, 256, 192, 128, 96, 64, Const::min_layer};
    // vector_s layer_sizes1{Const::max_layer, 200, 200, 200, Const::min_layer};
    
    // return 0;


//     {784, 389, 193, 96, 47, 26}
// Test 1 lr = 0.008; lrk = 0.97: 

    // fp_type lr = 0.007;
    // fp_type lrk = 0.98;
    // ConstT<fp_type>::SiLU_k1 = 1.0;

    for (fp_type lr : std::vector<fp_type>{0.007, 0.006}) {
        for (fp_type lrk : std::vector<fp_type>{1.0}) {
        for (size_vector layer_sizes1 : std::vector<size_vector>{
            // size_vector{Const::max_layer, 256, 192, 128, 96, 64, Const::min_layer},
                            // size_vector{Const::max_layer, 384, 288, 192, 144, Const::min_layer},
                            size_vector{Const::max_layer, 200, 150, 100, 50, Const::min_layer}}) {
 
            Const::lr_epoch_k = lrk;
            
            std::cout << "Test " << ++count << ' ' << "lr = " << lr << "; lr_epoch_k = "
                << Const::lr_epoch_k  << "; Const::lr_layers_k = " << Const::lr_layers_k << " | ";
            // for (auto ls : layer_sizes) std::cout << ls << ' ';
            SStr::Print(layer_sizes1);
            std::cout << '\n';

            MatrixModel MM(layer_sizes1, lr);

            auto time_point = s21::Time::Now();
            {

            DataManager letters_train("../datasets/emnist-letters-train.csv", -1, 1);
            // std::cout << "-------------------------------";
            // Letters letters_train("../datasets/a_z_data.csv");
            
            

            // letters_train.Expand();
            letters_train.Split(CROSS);
            letters_train.Shuffle();

            std::cout << "Dataset time = " << s21::Time::Duration(time_point) << '\n';

            // return 0;
            fp_type mean_ac = 0.0;

            time_point = s21::Time::Now();

            // for (int c = 0; c < CROSS; ++c) {

                // MatrixModel MM(layer_sizes1, lr);
                // std::cout << "Constructor time = " << s21::Time::Duration(time_point) << '\n';

                MM.Learn(letters_train, 5);
            //     auto metr = MM.Test(letters_train);
            //     mean_ac += metr.accuracy;
            //     std::cout << "CROSS accurecy = " << metr.accuracy << '\n';

            //     letters_train.CrossUpdate();
                
            // }

            // std::cout << "mean accuracy = " << mean_ac / (fp_type)CROSS << '\n';

            std::cout << "Learn time = " << s21::Time::Duration(time_point) << "\n\n";
            }

            {
            time_point = s21::Time::Now();
            DataManager letters_test1("../datasets/emnist-letters-test.csv", -1, 1);
            auto metrics = MM.Test(letters_test1);
            std::cout << metrics.accuracy << '\n';
            // s21::SStr::Print(metrics.precision, metrics.recall, metrics.F1);
            std::cout << "Test time = " << s21::Time::Duration(time_point) << "\n\n";
            }
            {
            time_point = s21::Time::Now();
            DataManager letters_test2("../datasets/emnist-letters-train.csv", -1, 1);
            auto metrics1 = MM.Test(letters_test2);
            std::cout << metrics1.accuracy << '\n';
            // s21::SStr::Print(metrics1.precision, metrics1.recall, metrics1.F1);
            std::cout << "Test time = " << s21::Time::Duration(time_point) << "\n\n";
            }
            {
            time_point = s21::Time::Now();
            DataManager letters_test3("../datasets/a_z_data.csv");
            auto metrics2 = MM.Test(letters_test3);
            s21::SStr::Print(metrics2.accuracy, '\n');
            // s21::SStr::Print(metrics2.precision, metrics2.recall, metrics2.F1);
            std::cout << "Test time = " << s21::Time::Duration(time_point) << "\n\n\n";
            }
        }
        }
    }

    return 0;
}


