#ifndef NEURAL_MATRIX_H_
#define NEURAL_MATRIX_H_

#include <vector>
#include <functional>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>

#include <gsl/gsl_blas.h>

namespace neural {
    class Matrix {
        double *_data;
        int _rows;
        int _cols;

    public:
        Matrix(int rows, int cols);
        Matrix(int rows, int cols, double val);
        Matrix(int rows, int cols, double data[]);
        Matrix(int rows, int cols, std::vector<double> data);
        Matrix(std::vector<std::vector<double>> data);

        /**
         * Copy and move semantics
         */
        Matrix(const Matrix &rhs);
        Matrix(Matrix &&rhs);
        Matrix &operator=(const Matrix &rhs);
        Matrix &operator=(Matrix &&rhs);
        
        ~Matrix();
        
        /**
         * Addition operator
         */
        const Matrix operator+(const Matrix &rhs) const;

        /**
         * Subtraction operator
         */
        const Matrix operator-(const Matrix &rhs) const;

        /**
         * Matrix multiplication
         */
        const Matrix operator*(const Matrix &rhs) const;

        /**
         * Scale operator
         */
        const Matrix operator*(const double factor) const;

        /**
         * Division operator
         */
        const Matrix operator/(const double factor) const;

        /**
         * Hadamard operator
         */
        const Matrix operator^(const Matrix &rhs) const;

        /**
         * In-place addition operator
         */
        void operator+=(const Matrix &rhs);

        /**
         * In-place subtraction operator
         */
        void operator-=(const Matrix &rhs);

        /**
         * In-place matrix multiplication
         */
        void operator*=(const Matrix &rhs);

        /**
         * In-place scale operator
         */
        void operator*=(const double factor);

        /**
         * In-place division operator
         */
        void operator/=(const double factor);

        /**
         * In-place hadamard operator
         */
        void operator^=(const Matrix &rhs);

        /**
         * Call a function on each item of the matrix
         */
        void map(std::function<double(double&)> function);

        /**
         * Get the transpose of the matrix
         */
        Matrix transpose() const;

        /**
         * Calculate the Euclidean norm of the matrix
         */
        double norm();

        /**
         * Make all values zero
         */
        void zero();

        /**
         * Fetch a value from the matrix
         */
        inline double get_at(const int row, const int col) const {
            return _data[row * _cols + col];
        }

        /**
         * Set a value in the matrix
         */
        inline void set_at(const int row, const int col, const double value) {
            _data[row * _cols + col] = value;
        }

        /**
         * Get the number of rows in the matrix;
         */
        inline int get_rows() const {
            return _rows;
        }

        /**
         * Get the number of columns in the matrix
         */
        inline int get_cols() const {
            return _cols;
        }

        /**
         * Get the pointer to the matrix data
         */
        inline double *data() const {
            return _data;
        }

        /**
         * Pretty print the matrix for debugging
         */
        void print() const;
    };
}

#endif