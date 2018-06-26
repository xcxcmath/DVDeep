//
// Created by 김 범주 on 2018. 6. 26..
//

#include "../../include/optimizer/Optimizer.h"

namespace dvdeep {
    namespace optimizer {

        Optimizer::Optimizer(network::FFN *net, double learning_rate)
            :m_net(net)
        {
            m_hp[HyperParamKey::lr] = learning_rate;
        }

        Optimizer::Optimizer(network::FFN *net, const HyperParam &hp, const Avg &avg)
            :m_net(net),
             m_hp(hp),
             m_avg(avg)
        {

        }

        double Optimizer::learn(const Matrix &x, const Matrix &sol) {
            m_in = x; m_sol = sol;
            auto lg = m_net->learn(x, sol);
            auto dv = getUpdate(lg.gradient);
            m_net->update(dv);
            return lg.loss;
        }

        HyperParam Optimizer::getHyperParameters() const noexcept {
            return m_hp;
        }

        Avg Optimizer::getAverage() const noexcept {
            return m_avg;
        }

        ParamVector Optimizer::getUpdate(const ParamVector &grad) {
            return getGradientStep(grad);
        }

        ParamVector Optimizer::getGradientStep(const ParamVector &grad) {
            ParamVector ret;
            for(const auto &param: grad){
                Param here;
                for(const auto &[key, val]:param){
                    here[key] = val * m_hp[HyperParamKey::lr]*-1.;
                }
                ret.push_back(here);
            }

            return ret;
        }

    }
}