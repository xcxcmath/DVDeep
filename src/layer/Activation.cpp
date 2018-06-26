//
// Created by bjk on 18. 6. 24.
//

#include "../../include/layer/Activation.h"

namespace dvdeep {
    namespace layer {


        ActFuncSet::ActFuncSet(const ScalarFunction &func, const ScalarFunction &df)
            :f(func), derivative(df) {}

        // The followings are defined for the definition of ActFuncSet objects
        // ReLU
        double f_relu(const double &x){
            return x * (x>0);
        }

        double d_relu(const double &x){
            return static_cast<double>(x>0);
        }

        // Logistic
        double f_logistic(const double &x){
            return 1. / (1.+std::exp(-x));
        };
        double d_logistic(const double &x){
            const double exp = f_logistic(x);
            return exp * (1. - exp);
        };

        // Hyperbolic Tangent
        double f_tanh(const double &x){
            return std::tanh(x);
        };
        double d_tanh(const double &x){
            const double tanh = f_tanh(x);
            return 1. - tanh*tanh;
        };


        // Activation Functions
        // TODO : solution for clang-tidy warnings..
        const ActFuncSet ReLU(f_relu, d_relu);
        const ActFuncSet Logistic(f_logistic, d_logistic);
        const ActFuncSet Tanh(f_tanh, d_tanh);


        Activation::Activation(const ActFuncSet &functions)
                : m_functions(functions) {}

        Matrix Activation::predict(const Matrix &x) {
            return x.unaryExpr(m_functions.f);
        }

        BackOutput Activation::backward(const Matrix &delta) {
            m_backout.delta = m_in.unaryExpr(m_functions.derivative);

            for(uint i = 0; i < delta.row; ++i)
                for(uint j = 0; j < delta.col; ++j)
                    m_backout.delta(i, j) *= delta(i, j);
            return m_backout;
        }

    }
}