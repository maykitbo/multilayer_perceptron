#include "test.h"

using namespace s21;

void CompareFiles(const std::string &f1, const std::string &f2) {
    std::ifstream file1(f1);
    std::ifstream file2(f2);
    std::string l1, l2;
    while(std::getline(file1, l1) && std::getline(file2, l2)) {
        if (l1 == "empty biases") {
            EXPECT_EQ(l2, "biases:");
            std::getline(file2, l2);
        } else if (l2 == "empty biases") {
            EXPECT_EQ(l1, "biases:");
            std::getline(file1, l1);
        } else {
            EXPECT_EQ(l1, l2);
        }
    }
}

template<class Model>
void OneModel() {
    {
        Model M(CorrectSettings());
        M.ToFile("tmp.prcp");
    }
    {
        Model M("tmp.prcp");
        M.ToFile("tmp2.prcp");
    }
    char c1, c2;
    std::ifstream file1("tmp.prcp");
    std::ifstream file2("tmp2.prcp");
    while (file1.get(c1) && file2.get(c2)) {
        EXPECT_EQ(c1, c2);
    }
    file1.close();
    file2.close();
    std::remove("tmp.prcp");
    std::remove("tmp2.prcp");
}

void IncorrectFileTest(const std::string &file_name) {
    EXPECT_ANY_THROW(GraphModel GM(file_name));
    EXPECT_ANY_THROW(MatrixModel MM(file_name));
}

TEST(File, matrix_model) {
    OneModel<MatrixModel>();
}

TEST(File, graph_model) {
    OneModel<GraphModel>();
}

TEST(File, universal) {
    {
        GraphModel GM(CorrectSettings());
        MatrixModel MM(CorrectSettings());
        GM.ToFile("tmp_graph.prcp");
        MM.ToFile("tmp_matrix.prcp");
    }
    {
        GraphModel GM("tmp_matrix.prcp");
        MatrixModel MM("tmp_graph.prcp");
        GM.ToFile("tmp_graph2.prcp");
        MM.ToFile("tmp_matrix2.prcp");
    }
    std::ifstream file_graph("tmp_graph.prcp");
    std::ifstream file_matrix("tmp_matrix.prcp");
    std::ifstream file_graph2("tmp_graph2.prcp");
    std::ifstream file_matrix2("tmp_matrix2.prcp");
    CompareFiles("tmp_graph.prcp", "tmp_matrix2.prcp");
    CompareFiles("tmp_matrix.prcp", "tmp_graph2.prcp");
    file_graph.close();
    file_matrix.close();
    file_graph2.close();
    file_matrix2.close();
    std::remove("tmp_graph.prcp");
    std::remove("tmp_matrix.prcp");
    std::remove("tmp_graph2.prcp");
    std::remove("tmp_matrix2.prcp");
}

TEST(File, error) {
    IncorrectFileTest("123");
    IncorrectFileTest(rp("/materials/bad_file_1.prcp"));
    IncorrectFileTest(rp("/materials/bad_file_2.prcp"));
    IncorrectFileTest(rp("/materials/bad_file_3.prcp"));
    IncorrectFileTest(rp("/materials/empty_file.prcp"));
}