//
// Created by bjk on 18. 6. 22.
//

#ifndef DVDEEP_FFN_H
#define DVDEEP_FFN_H

#include "../layer/Layer.h"

namespace dvdeep {
    namespace network {
        using Matrix = math::Matrix;

        class FFN {
        public:
            using ParamVector = std::vector<layer::Param>;
            struct LossGradient {
                double loss;
                ParamVector gradient;
            };

            //TODO: I CAN'T BE SURE TO USE SHARED_PTR
            using spLayer = std::shared_ptr<layer::Layer>;

            explicit FFN();
            virtual ~FFN();

            virtual Matrix predict(const Matrix &x);
            virtual LossGradient learn(const Matrix &x, const Matrix &sol);
            virtual void update(const ParamVector &pv);

            virtual void insert(layer::Layer *layer);
            virtual layer::Layer& operator[](size_t index);

        protected:
            std::vector<spLayer> m_layers;
        };
    }
}

#endif //DVDEEP_FFN_H
