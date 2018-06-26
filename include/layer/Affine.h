//
// Created by qawbecrdtey on 2018-06-22.
//
#include "Layer.h"
#ifndef DVDEEP_AFFINELAYER_H
#define DVDEEP_AFFINELAYER_H

namespace dvdeep {
    namespace layer {
        class Affine : public Layer {
            Affine() = default;

            Affine(uint input, uint output);

            ~Affine() override = default;

            Matrix predict(const Matrix &x) override;

            Matrix forward(const Matrix &x) override;

            BackOutput backward(const Matrix &delta) override;

            double getLoss(const Matrix &answer) const override;

        };
    }
}

#endif //DVDEEP_AFFINELAYER_H
