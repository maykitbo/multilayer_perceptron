#pragma once

// #include "../lib/matrix.h"
#include "../lib/blas_matrix.h"
// #include "../lib/cl_matrix.h"

namespace s21 {

typedef double fp_type;

typedef std::vector<size_t> size_vector;
typedef Matrix<fp_type> matrix_t;
typedef std::vector<fp_type> data_vector;
typedef std::vector<std::vector<data_vector>> dataset_t;

} // namespace s21



