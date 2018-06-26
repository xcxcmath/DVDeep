//
// Created by qawbecrdtey on 2018-06-19.
//
#include <cstring>
#include <algorithm>
#include "../../include/math/Matrix.h"

dvdeep::math::Matrix::~Matrix() {
    delete[] mat;
}

dvdeep::math::Matrix::Matrix()
        :row(0),col(0),mat(nullptr)
{

}

dvdeep::math::Matrix::Matrix(uint row, uint col)
        :row(row),col(col),mat(new double[row*col]())
{

}

dvdeep::math::Matrix::Matrix(uint row, uint col, double val)
        :row(row),col(col),mat(new double[row*col])
{
    for(uint i=0;i<row*col;i++)
        mat[i]=val;
}

template<typename InputIterator>
dvdeep::math::Matrix::Matrix(uint row, uint col, InputIterator it)
        :row(row),col(col),mat(new double[row*col])
{
    std::copy_n(it,row*col,mat);
}

dvdeep::math::Matrix::Matrix(const dvdeep::math::Matrix &A)
        :row(A.row),col(A.col),mat(new double[A.row*A.col])
{
    std::copy(A.mat,A.mat+row*col,mat);
}

dvdeep::math::Matrix::Matrix(dvdeep::math::Matrix &&A) noexcept
        :row(A.row),col(A.col),mat(A.mat)
{
    A.row=A.col=0;
    A.mat=nullptr;
}

dvdeep::math::Matrix &dvdeep::math::Matrix::operator=(const dvdeep::math::Matrix &A)
{
    row=A.row;
    col=A.col;
    mat=new double[row*col];

    std::copy(A.mat,A.mat+row*col,mat);
    return *this;
}

dvdeep::math::Matrix &dvdeep::math::Matrix::operator=(dvdeep::math::Matrix &&A) noexcept {
    row = A.row;
    col = A.col;
    mat = A.mat;

    A.row = A.col = 0;
    A.mat = nullptr;
    return *this;
}

dvdeep::math::Matrix::Matrix(uint row, uint col, double *mat)
        :row(row),col(col),mat(new double[row*col])
{
    std::copy_n(mat,row*col,this->mat);
}
