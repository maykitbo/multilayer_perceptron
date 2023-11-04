#pragma once

#include "../lib/matrix.h"

namespace s21 {

/**
 * @typedef fp_type
 * @brief Type alias for the floating-point number type used throughout the application (e.g., double).
 */
typedef double fp_type;

/**
 * @typedef size_vector
 * @brief Type alias for a vector of size_t values used to represent sizes or dimensions.
 */
typedef std::vector<size_t> size_vector;

/**
 * @typedef matrix_t
 * @brief Type alias for a matrix of floating-point values (e.g., a 2D array).
 */
typedef Matrix<fp_type> matrix_t;

/**
 * @typedef data_vector
 * @brief Type alias for a vector of floating-point values used to represent data points.
 */
typedef std::vector<fp_type> data_vector;

/**
 * @typedef data_vector
 * @brief Type alias for a vector of floating-point values used to represent data points.
 */
typedef std::pair<data_vector, int> data_t;

/**
 * @typedef dataset_t
 * @brief Type alias for vector of data_vector used to represent a dataset.
 */
// typedef std::vector<std::vector<data_vector>> dataset_t;
typedef std::vector<data_t> dataset_t;

} // namespace s21


