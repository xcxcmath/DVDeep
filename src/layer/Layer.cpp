//
// Created by bjk on 18. 6. 22.
//

#include "../../include/layer/Layer.h"

namespace dvdeep {
    namespace layer {
        Layer::Layer() = default;

        Layer::Layer(const Param &param)
                : m_param(param) {}

        Layer::~Layer() = default;

        Matrix dvdeep::layer::Layer::forward(const Matrix &x) {
            m_in = x;
            return m_out = predict(x);
        }

        void Layer::update(const Param &param) {
            for(const auto&[key, val] : param)
                m_param[key] = m_param[key] + val;
        }

        Param Layer::getParameters() const noexcept {
            return m_param;
        }

        double Layer::getLoss(const Matrix &answer) const {
            return 0.; // it should be non-zero if layer is for output!
        }
    }
}


