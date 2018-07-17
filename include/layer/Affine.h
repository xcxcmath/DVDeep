//
// Created by qawbecrdtey on 2018-06-22.
//
#include "Layer.h"
#ifndef DVDEEP_AFFINELAYER_H
#define DVDEEP_AFFINELAYER_H

namespace dvdeep {
    namespace layer {
        class Affine : public Layer {
        public:
            Affine(uint input, uint output);

            Matrix predict(const Matrix &x) override;

            BackOutput backward(const Matrix &delta) override;
        };
    }
}

#endif //DVDEEP_AFFINELAYER_H
