//
// Created by bjk on 18. 6. 22.
//

#ifndef DVDEEP_LAYER_H
#define DVDEEP_LAYER_H

#include "../Common.h"
#include "../math/Matrix.h"

namespace dvdeep {
    namespace layer {
        using Matrix = math::Matrix;

        enum class ParamKey {
            //common without Output
            initialized,

            //Affine
            weight, bias,
            init_stddev,

            //Dropout
            ratio,

            //BatchNorm
            beta, gamma, mean, var, momentum_eps,
        };
        using Param = std::map<ParamKey, Matrix>;

        struct BackOutput {
            Matrix delta;
            Param gradient;
        };

        class Layer {
        public:
            explicit Layer();
            explicit Layer(const Param &param);
            Layer(Param &&param) = delete;

            virtual ~Layer();

            virtual Matrix predict(const Matrix &x) = 0;
            virtual Matrix forward(const Matrix &x);
            virtual BackOutput backward(const Matrix &delta) = 0;

            virtual void update(const Param &param) final;

            Param getParameters() const noexcept;
            virtual double getLoss(const Matrix &answer) const;

        protected:
            Matrix m_in;
            Matrix m_out;
            BackOutput m_backout;
            Param m_param; //empty if layer is for output!
        };
    }
}

#endif //DVDEEP_LAYER_H
