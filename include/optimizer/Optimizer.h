//
// Created by 김 범주 on 2018. 6. 26..
//

#ifndef DVDEEP_OPTIMIZER_H
#define DVDEEP_OPTIMIZER_H

#include "../network/FFN.h"

namespace dvdeep {
    namespace optimizer {
        using Matrix = math::Matrix;
        using Param = layer::Param;
        using ParamVector = network::FFN::ParamVector ;

        enum class HyperParamKey {
            lr, gamma, beta1, beta2, eps, time, time_max,
        };
        using HyperParam = std::map<HyperParamKey, double>;

        enum class AvgKey {
            first, second,
        };
        using Avg = std::map<AvgKey, ParamVector>;

        class Optimizer {
        public:
            explicit Optimizer(network::FFN *net, double learning_rate = 0.01);
            explicit Optimizer(network::FFN *net,
                                const HyperParam &hp,
                                const Avg &avg);

            virtual double learn(const Matrix &x, const Matrix &sol);

            HyperParam getHyperParameters() const noexcept;
            Avg getAverage() const noexcept;

        protected:
            virtual ParamVector getUpdate(const ParamVector &grad);
            ParamVector getGradientStep(const ParamVector &grad);

            network::FFN *m_net;

            HyperParam m_hp;
            Avg m_avg;

            Matrix m_in;
            Matrix m_sol;
        };
    }
}

#endif //DVDEEP_OPTIMIZER_H
