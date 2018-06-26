//
// Created by bjk on 18. 6. 24.
//

#ifndef DVDEEP_ACTIVATION_H
#define DVDEEP_ACTIVATION_H

#include "Layer.h"

namespace dvdeep {
    namespace layer {
        using ScalarFunction = std::function<double(const double &)>;

        struct ActFuncSet {
            ScalarFunction f;           // forward
            ScalarFunction derivative;  // backward

            ActFuncSet(const ScalarFunction &func, const ScalarFunction &df);
        };

        extern const ActFuncSet ReLU;
        extern const ActFuncSet Logistic;
        extern const ActFuncSet Tanh;

        class Activation : public Layer {
        public:
            explicit Activation(const ActFuncSet &functions);

            Matrix predict(const Matrix &x) override;
            BackOutput backward(const Matrix &delta) override;

        protected:
            ActFuncSet m_functions;
        };
    }
}

#endif //DVDEEP_ACTIVATION_H
