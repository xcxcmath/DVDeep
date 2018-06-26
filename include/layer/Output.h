//
// Created by 김 범주 on 2018. 6. 26..
//

#ifndef DVDEEP_OUTPUT_H
#define DVDEEP_OUTPUT_H

#include "Layer.h"

namespace dvdeep {
    namespace layer {
        using OutputFunction = std::function<Matrix(const Matrix&)>;
        using LossFunction = std::function<double(const Matrix&, const Matrix&)>;
        struct OutputFuncSet {
            OutputFunction f;
            LossFunction loss;
            OutputFuncSet (const OutputFunction &out, const LossFunction &loss_func);
        };

        extern const OutputFuncSet Softmax;
        extern const OutputFuncSet Identity;

        class Output : public Layer {
        public:
            explicit Output(const OutputFuncSet &functions);

            Matrix predict(const Matrix &x) override;
            BackOutput backward(const Matrix &delta) override;

            double getLoss(const Matrix &sol) const override;
        protected:
            OutputFuncSet m_functions;
        };
    }
}

#endif //DVDEEP_OUTPUT_H
