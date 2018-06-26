#include <iostream>
#include "include/math/Matrix.h"
using namespace dvdeep::math;
double func(double x)
{
    return x * (1 - x);
}
int main() {
    std::cout << "Hello, World!" << std::endl;
    double a[6]={1,2,3,4,5,6};
    double b[9]={1,2,3,4,5,6,7,8,9};
    Matrix A=Matrix(2,3,a);
    Matrix B=Matrix(3,3,b);
    std::cout<<A*B<<std::endl;
    std::cout<<A.unaryExpr(func)<<std::endl;
    return 0;
}