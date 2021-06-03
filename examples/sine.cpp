#include <iostream>
#include <cmath>
#include "../src/neural.h"

int main() {
    const double pi = 3.14159265358979323846;
    const int population_size = 360;
    
    std::vector<neural::DataSample> examples;
    for(int j = 0; j < population_size; j++) {
        double x = j * pi/180;
        double y = sin(x);
        examples.push_back({{x}, {y}});
    }
    neural::NetworkParameters params;
    params.learning_rate = 0.75;
    params.gradient_clip = 1.0;
    params.cost_function = neural::quadratic_cost;
    params.layers = {
        {1, nullptr},
        {10, neural::lrelu},
        {10, neural::lrelu},
        {10, neural::lrelu},
        {1, neural::tanh},
    };
    neural::Network network(params);
    int epochs = 10000;
    for(int i = 0; i < epochs; i++) {
        network.fit(examples, 12);
    }
    for(auto &ex : examples) {
        neural::Matrix output = network.forward(ex.input);
        std::cout << ex.input[0] << " " << output.get_at(0, 0) << " " << ex.output[0] << "\n";
    }
    return 0;
}