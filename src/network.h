#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include <vector>
#include <random>
#include <fstream>

#include "matrix.h"
#include "costs.h"

namespace neural {
    // Activation/cost functions must take a double and a boolean indicating
    // whether or not it's the derivative function
    using activation = double (*)(double x, bool derivative);
    using cost = double (*)(double predicted_y, double expected_y, bool derivative);

    struct Layer {
        int nodes;
        activation activation_function;
    };

    struct NetworkParameters {
        cost cost_function   = quadratic_cost;
        double learning_rate = 1.0;
        double gradient_clip = 1.0;
        std::vector<Layer> layers;
    };

    struct DataSample {
        std::vector<double> input;
        std::vector<double> output;
    };

    class Network {
        std::vector<Matrix> _z;
        std::vector<Matrix> _a;
        std::vector<Matrix> _biases;
        std::vector<Matrix> _weights;

        std::vector<Matrix> _weight_deltas;
        std::vector<Matrix> _bias_deltas;

        std::vector<activation> _activations;
        cost _cost;

        double _learning_rate;
        double _gradient_clip;

        /**
         * Apply the cost function to matrices
         */
        Matrix apply_cost(Matrix &predicted, Matrix &expected, bool derivative);

        /**
         * Update the network parameters using the gradient averages (minibatch)
         */
        void update_parameters(int m);

    public:
        Network(NetworkParameters params);

        /**
         * Feed-forward algorithm
         */
        Matrix forward(std::vector<double> input);

        /**
         * Train network on a set of sample data in batches of
         * size m using the backpropagation algorithm
         */
        void fit(std::vector<DataSample> samples, int m=1);

        /**
         * Write the neural network state (weights and biases) to disk
         */
        void save(std::string filename);

        /**
         * Load the neural network state (weights and biases) from disk
         * 
         * The number of parameters at each layer should match that
         * of the constructed network
         */
        void load(std::string filename);
    };
}

#endif