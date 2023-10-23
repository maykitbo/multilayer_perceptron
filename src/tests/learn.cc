#include "test.h"

using namespace s21;

template<class Model>
void Testl() {
    auto ps = CorrectSettings();
    ps.layers = {784, 128, 100, 84, 26};
    Model MM(CorrectSettings());
    DataManager data(rp("datasets/emnist-letters-train.csv"), -1, k90Rotate);
    data.Split(5);
    bool stop = false;
    auto m1 = MM.Test(data);
    MM.Learn(data, 2, stop, false, false);
    auto m2 = MM.Test(data);
    EXPECT_EQ(m1.accuracy < m2.accuracy, true);
}

TEST(Learn, matrix_model) {
    Testl<MatrixModel>();
}

TEST(Learn, graph_model) {
    Testl<GraphModel>();
}

TEST(Dataset, incorrect_files) {
    EXPECT_ANY_THROW(DataManager data(rp("materials/test_dataset/bad_1.txt"), -1, k90Rotate));
    EXPECT_ANY_THROW(DataManager data(rp("materials/test_dataset/bad_2.txt"), 0, k90Rotate));
    EXPECT_ANY_THROW(DataManager data(rp("materials/test_dataset/bad_3.txt"), 0, k90Rotate));
}

TEST(Dataset, correct_file) {
    EXPECT_NO_THROW(DataManager data(rp("materials/test_dataset/file.txt"), -1, k90Rotate));
}

