#pragma once

#include <iostream>
#include <functional>
#include <initializer_list>
#include <random>

template<class T>
class Matrix {
    T **matrix_;
    int rows_, cols_;
    void create(std::function<void(int, int)> func) {
        matrix_ = new T *[rows_];
        for (int k = 0; k < rows_; ++k) {
            matrix_[k] = new T [cols_];
            for (int g = 0; g < cols_; ++g) {
                func(k, g);
            }
        }
    }

    public:
        Matrix() noexcept : rows_(0), cols_(0) {}
        Matrix(int rows, int cols) : rows_(rows), cols_(cols) {
            matrix_ = new T *[rows_];
            for (int k = 0; k < rows_; ++k) {
                matrix_[k] = new T [cols_];
            }
        }
        Matrix(int rows, int cols, const T value) : rows_(rows), cols_(cols) {
            create([&] (int k, int g) { matrix_[k][g] = value; });
        }
        Matrix(int rows, int cols, std::function<const T(void)> value_func) : rows_(rows), cols_(cols) {
            create([&] (int k, int g) { matrix_[k][g] = value_func(); });
        }
        Matrix(int rows, int cols, std::function<void(T&)> value_func) : rows_(rows), cols_(cols) {
            create([&] (int k, int g) { value_func(matrix_[k][g]); });
        }
        Matrix(int rows, int cols, std::function<void(T&, int, int)> value_func) : rows_(rows), cols_(cols) {
            create([&] (int k, int g) { value_func(matrix_[k][g], k, g); });
        }
        Matrix(int rows, int cols, T mean, T sd) : rows_(rows), cols_(cols) { // only for float, double, long double
            std::mt19937 generator{std::random_device{}()};
            std::normal_distribution<T> distribution{mean, sd};
            create([&] (int k, int g) { matrix_[k][g] = distribution(generator); });
        }
        Matrix(const Matrix &other) : rows_(other.rows_), cols_(other.cols_) {
            create([&] (int k, int g) { matrix_[k][g] = other(k, g); });
        }
        Matrix(Matrix &&other) noexcept : rows_(other.rows_), cols_(other.cols_), matrix_(other.matrix_) {
            other.matrix_ = nullptr;
            other.rows_ = 0;
            other.cols_ = 0;
        }
        void Loop(std::function<void(int, int)> func) {
            for (int k = 0; k < rows_; ++k) {
                for (int g = 0; g < cols_; ++g) {
                    func(k, g);
                }
            }
        }
        Matrix& operator=(const Matrix& other) {
            if (this != &other) {
                Matrix temp(other);
                std::swap(rows_, temp.rows_);
                std::swap(cols_, temp.cols_);
                std::swap(matrix_, temp.matrix_);
            }
            return *this;
        }
        Matrix& operator=(Matrix&& other) noexcept {
            std::swap(rows_, other.rows_);
            std::swap(cols_, other.cols_);
            std::swap(matrix_, other.matrix_);
            return *this;
        }
        void Clear() {
            if (cols_ > 0) {
                for (int k = 0; k < rows_; ++k) {
                    delete[] matrix_[k];
                }
            }
            if (rows_ > 0) delete[] matrix_;
            cols_ = 0;
            rows_ = 0;
        }
        ~Matrix() {
            Clear();
        }
        Matrix(std::initializer_list<std::initializer_list<T>> const &items) : rows_(items.size()), cols_(items.begin()->size()) {
            int k = 0;
            matrix_ = new T *[rows_];
            for (auto const &row : items) {
                int g = 0;
                matrix_[k] = new T [cols_];
                for (auto const &cell : row) {
                    matrix_[k][g++] = cell;
                }
                ++k;
            }
        }
        void ForEach(std::function<void(const T&)> func) const {
            loop([&] (int k, int g) { func(matrix_[k][g]); });
        }
        void Fill(std::function<void(T&)> func) {
            loop([&] (int k, int g) { func(matrix_[k][g]); });
        }
        T &operator()(int row, int col) {
            return matrix_[row][col];
        }
        const T &operator()(int row, int col) const {
            return matrix_[row][col];
        }
        const T &Get(int row, int col) const {
            return matrix_[row][col];
        }
        void Set(int row, int col, T value) {
            if (row >= rows_ || row < 0 || col >= cols_ || col < 0) {
                throw std::invalid_argument("Matrix: Set error");
            }
            matrix_[row][col] = value;
        }
        int GetRows() const {
            return rows_;
        }
        // T *GetRow(int row) const {
        //     return matrix_[row];
        // }
        int GetCols() const {
            return cols_;
        }
        Matrix operator*(const T num) const {
            return Matrix(rows_, cols_, [&] (T &cell, int k, int g) { cell = matrix_[k][g] * num; });
        }
        void operator*=(const T num) {
            loop([&] (int k, int g) { matrix_[k][g] *= num; });
        }
        Matrix operator*(Matrix &other) const {
            if (cols_ != other.rows_) {
                throw std::invalid_argument("Matrix: Mul error");
            }
            return Matrix(rows_, other.cols_, [&] (T &cell, int k, int g) {
                cell = 0;
                for (int i = 0; i < cols_; ++i) {
                    cell += matrix_[k][i] * other(i, g);
                }
            });
        }
        void operator*=(Matrix &other) {
            auto temp = *this;
            *this = temp * other;
        }
        Matrix operator+(Matrix &other) const {
            if (rows_ != other.rows_ || cols_ != other.cols_) {
                throw std::invalid_argument("Matrix: Sum error");
            }
            return Matrix(rows_, cols_, [&] (T &cell, int k, int g) {
                cell = matrix_[k][g] + other(k, g);
            });
        }
        void operator+=(Matrix &other) {
            if (rows_ != other.rows_ || cols_ != other.cols_) {
                throw std::invalid_argument("Matrix: Sum error");
            }
            Loop([&] (int k, int g) { matrix_[k][g] += other(k, g); });
        }
        void operator+=(const Matrix &other) {
            if (rows_ != other.rows_ || cols_ != other.cols_) {
                throw std::invalid_argument("Matrix: Sum error");
            }
            Loop([&] (int k, int g) { matrix_[k][g] += other(k, g); });
        }
        Matrix operator-(Matrix &other) const {
            if (rows_ != other.rows_ || cols_ != other.cols_) {
                throw std::invalid_argument("Matrix: Sum error");
            }
            return Matrix(rows_, cols_, [&] (T &cell, int k, int g) {
                cell = matrix_[k][g] - other(k, g);
            });
        }
        void operator-=(Matrix &other) {
            if (rows_ != other.rows_ || cols_ != other.cols_) {
                throw std::invalid_argument("Matrix: Sum error");
            }
            Loop([&] (int k, int g) { matrix_[k][g] -= other(k, g); });
        }
        void operator-=(const Matrix &other) {
            if (rows_ != other.rows_ || cols_ != other.cols_) {
                throw std::invalid_argument("Matrix: Sum error");
            }
            Loop([&] (int k, int g) { matrix_[k][g] -= other(k, g); });
        }
        Matrix Transpose() const {
            return Matrix(cols_, rows_, [&] (T &cell, int k, int g) {
                cell = matrix_[g][k];
            });
        }
        void TransposeThis() {
            auto temp = *this;
            *this = temp.Transpose();
        }
        bool operator==(const Matrix &other) const {
            if (cols_ != other.cols_ || rows_ != other.rows_) return false;
            for (int k = 0; k < rows_; ++k) {
                for (int g = 0; g < cols_; ++g) {
                    if (other(k, g) != matrix_[k][g]) {
                        return false;
                    }
                }
            }
            return true;
        }
        bool operator!=(Matrix &other) {
            return !operator==(other);
        }
        T Determinant() {
            if (rows_ != cols_) {
                throw std::invalid_argument("Matrix: Determinant error");
            }
            return determinant();
        }
        Matrix Minor(int row, int col) {
            Matrix res(rows_ - 1, cols_ - 1);
            int i = 0, j;
            for (int k = 0; k < rows_; k++) {
                j = 0;
                if (k == row) continue;
                for (int g = 0; g < cols_; g++) {
                if (g != col) {
                    res.matrix_[i][j] = matrix_[k][g];
                    j++;
                }
                }
                i++;
            }
            return res;
        }
        void Print() const {
            for (int k = 0; k < rows_; ++k) {
                for (int g = 0; g < cols_; ++g) {
                    std::cout << matrix_[k][g] << " ";
                }
                std::cout << "\n";
            }
        }
    private:
        T determinant() {
            T res = 0;
            if (rows_ == 1) {
                res = matrix_[0][0];
            } else {
                for (int k = 0; k < cols_; k++) {
                    res += Minor(0, k).Determinant() * matrix_[0][k] * (k % 2 ? -1 : 1);
                }
            }
            return res;
        }
};
