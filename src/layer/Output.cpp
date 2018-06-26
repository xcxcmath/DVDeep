//
// Created by 김 범주 on 2018. 6. 26..
//

#include "../../include/layer/Output.h"

namespace dvdeep {
    namespace layer {

        OutputFuncSet::OutputFuncSet(const OutputFunction &out, const LossFunction &loss_func)
            :f(out), loss(loss_func)
        {}

        // The followings are defined for the definition of OutputFuncSet objects
        double log(const double& x){
            return std::log(x);
        }
        double sq(const double& x){
            return x * x;
        }
        Matrix f_softmax(const Matrix &x){
            Matrix ret = x;
            for(uint j = 0; j < x.col; ++j){
                double max = x(0, j);
                double sum = 0;
                for(uint i = 1; i < x.row; ++i) {
                    max = std::max(max, x(i, j));
                }
                for(uint i = 0; i < x.row; ++i)
                    ret(i, j) -= max;
                for(uint i = 0; i < x.row; ++i) {
                    sum += ret(i, j);
                    ret(i, j) = std::exp(ret(i, j));
                }
                const double sum_exp = std::exp(sum);
                for(uint i = 0; i < x.row; ++i){
                    ret(i, j) /= sum_exp;
                }
            }

            return ret;
        }
        double cross_entropy(const Matrix &y, const Matrix &sol){
            double ret = 0.;
            Matrix log_y = y.unaryExpr(log);
            for(uint i = 0; i < y.row; ++i)
                for(uint j = 0; j < y.col; ++j)
                    ret -= sol(i, j) * log_y(i, j);
            return ret / y.col;
        }
        Matrix f_identity(const Matrix &x){
            return x;
        }
        double squared(const Matrix &y, const Matrix &sol){
            double ret = 0.;
            Matrix sq_y = (y-sol).unaryExpr(sq);
            for(uint i = 0; i < y.row; ++i)
                for(uint j = 0; j < y.col; ++j)
                    ret += sq_y(i, j);
            return ret / 2. / y.col;
        }

        const OutputFuncSet Softmax(f_softmax, cross_entropy);
        const OutputFuncSet Identity(f_identity, squared);

        Output::Output(const OutputFuncSet &functions)
            :m_functions(functions) {}

        Matrix Output::predict(const Matrix &x) {
            return m_functions.f(x);
        }

        BackOutput Output::backward(const Matrix &delta) {
            m_backout.delta = (m_out - delta) * (1./delta.col);
            return m_backout;
        }

        double Output::getLoss(const Matrix &sol) const {
            return m_functions.loss(m_out, sol);
        }
    }
}