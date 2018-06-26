//
// Created by bjk on 18. 6. 22.
//

#include "../../include/network/FFN.h"

namespace dvdeep {
    namespace network {

        FFN::FFN() = default;

        FFN::~FFN() = default;

        Matrix FFN::predict(const Matrix &x){
            auto ret = x;

            for(auto &layer: m_layers)
                ret = layer->predict(ret);

            return ret;
        }

        FFN::LossGradient FFN::learn(const Matrix &x, const Matrix &sol) {
            auto temp = x;
            for(auto &layer: m_layers)
                temp = layer->forward(temp);

            const auto loss = m_layers.back()->getLoss(sol);

            ParamVector gradient;

            temp = sol;

            for(auto it = m_layers.rbegin(); it != m_layers.rend();++it){
                const auto here = (*it)->backward(temp);
                temp = here.delta;
                gradient.push_back(here.gradient);
            }

            std::reverse(gradient.begin(), gradient.end());

            return {loss, gradient};
        }

        void FFN::update(const FFN::ParamVector &pv) {
            for(size_t i = 0; i < pv.size(); ++i)
                m_layers[i]->update(pv[i]);
        }

        void FFN::insert(layer::Layer *layer) {
            m_layers.emplace_back(layer);
        }

        layer::Layer &FFN::operator[](size_t index) {
            return *m_layers[index];
        }


    }
}
