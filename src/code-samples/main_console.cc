#include "model/data_manager.h"
#include "model/matrix_model.h"
#include "model/graph_model.h"
#include "lib/m_time.h"

#include "lib/random.h"

using namespace s21;



int main() {

    int CROSS = 5;


    int epochs = 3;

    int layers = 4;

    size_vector layer_sizes1{Const::max_layer, 100, 80, 50, Const::min_layer};
    fp_type learning_rate = 0.0075;

    PerceptronSettings ps;
    ps.SetActivation(kSiLU);
    ps.layers = layer_sizes1;
    ps.learning_rate = learning_rate;
    ps.SetWeightInit(kXavier);
    ps.momentum = 0.15;


    DataManager letters_train("../datasets/emnist-letters-train.csv", -1, k90Rotate);


    MatrixModel mm(ps);


    return 0;
}


