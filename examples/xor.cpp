#include <iostream>
#include "../src/neural.h"

int main() {
    std::vector<neural::DataSample> examples = {
        {{1.0, 0.0}, {1.0}},
        {{0.0, 1.0}, {1.0}},
        {{0.0, 0.0}, {0.0}},
        {{1.0, 1.0}, {0.0}},
    };
    neural::NetworkParameters params;
    params.learning_rate = 1.0;
    params.gradient_clip = 1.5;
    params.cost_function = neural::quadratic_cost;
    params.layers = {
        {2, nullptr},
        {2, neural::lrelu},
        {1, neural::sigmoid},
    };
    neural::Network network(params);
    for(int i = 0; i < 10000; i++) {
        network.fit(examples, 2);        
    }
    for(auto &ex : examples) {
        neural::Matrix output = network.forward(ex.input);
        for(auto &val : ex.input) {
            std::cout << int(val) << " ";
        }
        output.print();
    }
    return 0;
}