//
// Created by qawbecrdtey on 2018-06-22.
//

#include "../../include/layer/Affine.h"

dvdeep::layer::Affine::Affine(uint input, uint output) {
    m_param[ParamKey::weight] = dvdeep::math::Matrix(input, output);
    double sum = 0;
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1);
A:
    for(uint i = 0; i < input; i++)
        for(uint j = 0; j < output; j++)
            sum += (m_param[ParamKey::weight](i, j) = dis(gen));

    double mean = sum / (input * output);
    double var = 0;
    if(mean != 0)
        for(uint i = 0; i < input; i++)
            for(uint j = 0; j < output; j++)
                var += (m_param[ParamKey::weight](i, j) = (m_param[ParamKey::weight](i, j) - mean));

    double s = std::sqrt(var);
    if(s != 0)
        for (uint i = 0; i < input; i++)
            for (uint j = 0; j < output; j++)
                m_param[ParamKey::weight](i, j) /= s;
    else
        if(input * output != 1)
            goto A;
    m_param[ParamKey::bias] = dvdeep::math::Matrix(output, 1);
}

dvdeep::layer::Matrix dvdeep::layer::Affine::predict(const dvdeep::layer::Matrix &x) {
    return m_param.at(ParamKey::weight).transpose() * x + m_param.at(ParamKey::bias);
}

dvdeep::layer::BackOutput dvdeep::layer::Affine::backward(const dvdeep::layer::Matrix &delta) {
    return BackOutput();
}