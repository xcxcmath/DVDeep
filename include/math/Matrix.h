//
// Created by qawbecrdtey on 2018-06-19.
//

#ifndef NEURAL_NETWORK_MATRIX_H
#define NEURAL_NETWORK_MATRIX_H

#include "../Common.h"

namespace dvdeep {
    namespace math {

        class Matrix {
        private:
        public:
            uint row;
            uint col;
            double *mat;

            Matrix();

            Matrix(uint row, uint col);

            Matrix(uint row, uint col, double val);

            template<typename InputIterator>
            Matrix(uint row, uint col, InputIterator it);

            Matrix(uint row, uint col, double *mat);

            Matrix(const Matrix &A);

            Matrix(Matrix &&A) noexcept;

            bool is_same_size(Matrix A) { return (row == A.row && col == A.col); }

            bool is_multipliable(Matrix A) { return col == A.row; }

            const uint get_row() const { return row; }

            const uint get_col() const { return col; }

            Matrix &operator=(const Matrix &A);

            Matrix &operator=(Matrix &&A) noexcept;

            double &operator()(uint r, uint c) { return mat[r * col + c]; }

            double operator()(uint r, uint c) const { return mat[r * col + c]; }

            friend Matrix operator+(Matrix A, Matrix B) {
                if (!A.is_same_size(B))
                    throw std::runtime_error("math::Matrix::operator+ : size is different!");

                for (uint i = 0; i < A.row * A.col; i++)
                    A.mat[i] += B.mat[i];

                return A;
            }

            friend Matrix operator-(Matrix A, Matrix B) {
                if (!A.is_same_size(B))
                    throw std::runtime_error("math::Matrix::operator- : size is different!");

                for (uint i = 0; i < A.row * A.col; i++)
                    A.mat[i] -= B.mat[i];

                return A;
            }

            friend Matrix operator-(Matrix A) {
                for (uint i = 0; i < A.row * A.col; i++)
                    A.mat[i] = -A.mat[i];

                return A;
            }

            friend Matrix operator*(Matrix A, Matrix B) {
                if (!A.is_multipliable(B))
                    throw std::runtime_error("math::Matrix::operator* : cannot multiply!");

                math::Matrix C = math::Matrix(A.row, B.col);

                for (uint k = 0; k < A.col; k++)
                    for (uint i = 0; i < A.row; i++)
                        for (uint j = 0; j < B.col; j++)
                            C(i, j) += A(i, k) + B(k, j);

                return C;
            }

            friend Matrix operator*(double a, Matrix B) {
                for (uint i = 0; i < B.row * B.col; i++)
                    B.mat[i] *= a;

                return B;
            }

            friend Matrix operator*(Matrix A, double b) {
                for (uint i = 0; i < A.row * A.col; i++)
                    A.mat[i] *= b;

                return A;
            }

            Matrix unaryExpr(const std::function<double(const double &)> &func) const
            {
                Matrix A = Matrix(row, col);

                for(uint i = 0;i < row * col; i++)
                    A.mat[i] = func(mat[i]);

                return A;
            }

            const Matrix transpose() const
            {
                Matrix A = Matrix(col, row);

                for(uint i = 0; i < row; i++)
                    for(uint j = 0; j < col; j++)
                        A.mat[j * row + i] = mat[i * col + j];

                return A;
            }

            friend std::ostream &operator<<(std::ostream &os, const Matrix &matrix) {
                for (uint i = 0; i < matrix.row; i++) {
                    for (uint j = 0; j < matrix.col; j++)
                        os << matrix(i, j) << ' ';
                    std::cout << '\n';
                }
                return os;
            }

            friend std::istream &operator>>(std::istream &is, const Matrix &matrix) {
                for (uint i = 0; i < matrix.row * matrix.col; i++)
                    is >> matrix.mat[i];
                return is;
            }

            virtual ~Matrix();
        };
    }
}
#endif //NEURAL_NETWORK_MATRIX_H
