#include <iostream>
#include "include/layer/Affine.h"
#include "include/layer/Output.h"
#include "include/layer/Activation.h"
#include "include/optimizer/Optimizer.h"

using namespace dvdeep;

std::pair<math::Matrix, math::Matrix> makeBatch(uint batch){
    std::random_device rd;

    math::Matrix input(60, batch);
    math::Matrix output(2, batch);

    for(uint _i = 0; _i < batch; ++_i){

        // generate solution
        std::vector<int> numbers(10);
        std::iota(numbers.begin(), numbers.end(), 0); // 0 to 9
        std::shuffle(numbers.begin(), numbers.end(), rd);
        std::vector<int> sol(numbers.begin(), numbers.begin()+3);

        for(int i = 0; i < 3; ++i){
            input(30+i*10+sol[i], _i) = 1;
        }

        std::uniform_int_distribution pick_dist(0, 3);
        const int picked = pick_dist(rd);
        std::shuffle(sol.begin(), sol.end(), rd);
        std::vector<int> user(sol.begin(), sol.begin()+picked);
        for(int i = 0; i < 3-picked; ++i){
            user.push_back(numbers[i+3]);
        }

        std::shuffle(user.begin(), user.end(), rd);

        for(int i = 0; i < 3; ++i){
            input(i*10 + user[i], _i) = 1;
        }

        int strike = 0, ball = 0;
        for(int i = 0; i < 3; ++i){
            if(sol[i] == user[i]) ++strike;
            else if(std::find(sol.begin(), sol.end(), user[i]) != sol.end()){
                ++ball;
            }
        }

        output(0, _i) = strike;
        output(1, _i) = ball;
    }

    return std::make_pair(input, output);
}

int main() {
    network::FFN net;

    net.insert(new layer::Affine(60, 60));
    net.insert(new layer::Activation(layer::ReLU));
    net.insert(new layer::Affine(60, 2));
    net.insert(new layer::Output(layer::Identity));

    optimizer::Optimizer opt(&net);

    return 0;
}