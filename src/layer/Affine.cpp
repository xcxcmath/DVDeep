//
// Created by qawbecrdtey on 2018-06-22.
//

#include "../../include/layer/Affine.h"

dvdeep::layer::Matrix dvdeep::layer::Affine::predict(const dvdeep::layer::Matrix &x) {
    return m_param.at(ParamKey::weight).transpose() * x + m_param.at(ParamKey::bias);
}

dvdeep::layer::Matrix dvdeep::layer::Affine::forward(const dvdeep::layer::Matrix &x) {
    return Layer::forward(x);
}

dvdeep::layer::BackOutput dvdeep::layer::Affine::backward(const dvdeep::layer::Matrix &delta) {
    return BackOutput();
}

double dvdeep::layer::Affine::getLoss(const dvdeep::layer::Matrix &answer) const {
    uint length = answer.get_row();
    double loss = 0;
    for(uint i = 0; i < length; i++)
    {
        loss += (m_out(i, 0) - answer(i, 0)) * (m_out(i, 0) - answer(i, 0));
    }
    return loss;
}
