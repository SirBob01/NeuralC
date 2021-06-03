#include "network.h"

namespace neural {
    Network::Network(NetworkParameters params) {
        int n = params.layers.size();

        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution<double> distribution;
        
        for(int i = 0; i < n; i++) {
            _a.emplace_back(1, params.layers[i].nodes, 0.0);

            if(i < n - 1) {
                // Add weight matrix (i, i+1) and its delta
                int m = params.layers[i].nodes;
                int n = params.layers[i+1].nodes;
                _weights.emplace_back(m, n);
                _weight_deltas.emplace_back(m, n);

                // Kaiming initialization of weights
                double kaiming_factor = std::sqrt(2.0/m);
                for(int j = 0; j < m; j++) {
                    for(int k = 0; k < n; k++) {
                        double r = distribution(generator) * kaiming_factor;
                        _weights.back().set_at(j, k, r);
                    }
                }
            }
            if(i > 0) {
                _z.emplace_back(1, params.layers[i].nodes, 0.0);
                _activations.push_back(params.layers[i].activation_function);
            
                // Add bias matrix and its delta
                _bias_deltas.emplace_back(1, params.layers[i].nodes);
                _biases.emplace_back(1, params.layers[i].nodes);
            }
        }

        _cost = params.cost_function;
        _learning_rate = params.learning_rate;
        _gradient_clip = params.gradient_clip;
    }

    Matrix Network::apply_cost(Matrix &predicted, Matrix &expected, bool derivative) {
        int rows = expected.get_rows();
        int cols = expected.get_cols();
        Matrix loss(rows, cols);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                loss.set_at(i, j, _cost(predicted.get_at(i, j), expected.get_at(i, j), derivative));
            }
        }
        return loss;
    }

    void Network::update_parameters(int m) {
        double delta_factor = _learning_rate / m;
        for(int j = 0; j < _activations.size(); j++) {
            // Calculate gradient clipping and learning rate adjustments
            Matrix weight_grad = _weight_deltas[j] * delta_factor;
            double weight_grad_norm = weight_grad.norm();
            if(weight_grad_norm >_gradient_clip) {
                weight_grad *= _gradient_clip / weight_grad_norm;
            }

            Matrix bias_grad = _bias_deltas[j] * delta_factor;
            double bias_grad_norm = bias_grad.norm();
            if(bias_grad_norm >_gradient_clip) {
                bias_grad *= _gradient_clip / bias_grad_norm;
            }

            // Update parameters
            _weights[j] -= weight_grad;
            _biases[j] -= bias_grad;
            
            // Reset the deltas
            _weight_deltas[j].zero();
            _bias_deltas[j].zero();
        }
    }

    Matrix Network::forward(std::vector<double> input) {
        _a[0] = {1, _weights[0].get_rows(), input};

        int n = _weights.size();
        for(int i = 0; i < n; i++) {
            _z[i] = (_a[i] * _weights[i]) + _biases[i];
            _a[i+1] = _z[i];
            _a[i+1].map([this, i](double &n) {
                return _activations[i](n, false);
            });
        }
        return _a.back();
    }

    void Network::fit(std::vector<DataSample> samples, int m) {
        m = std::min(m, static_cast<int>(samples.size()));

        for(int i = 0; i < samples.size(); i++) {
            DataSample &sample = samples[i];
            forward(sample.input);
            
            Matrix expected = {1, _z.back().get_cols(), sample.output};
            Matrix delta_l = apply_cost(_a.back(), expected, true);
            
            // Backpropagation algorithm
            int n = _activations.size();
            for(int j = n - 1; j >= 0; j--) {
                if(j < n - 1) {
                    delta_l *= _weights[j+1].transpose();
                }
                _z[j].map([this, j](double &n) {
                    return _activations[j](n, true);
                });
                delta_l ^= _z[j];

                _weight_deltas[j] += _a[j].transpose() * delta_l;
                _bias_deltas[j] += delta_l;
            }

            // Update the network parameters in batches
            if((i + 1) % m == 0) {
                update_parameters(m);
            }
        }
        
        // Sample size is not divisble by m; train on final batch
        m = samples.size() % m;
        if(m) {
            update_parameters(m);
        }
    }

    void Network::save(std::string filename) {
        std::ofstream outfile;
        outfile.open(filename, std::ios::binary | std::ios::out);
        for(int i = 0; i < _activations.size(); i++) {
            Matrix &weight = _weights[i];
            Matrix &bias = _biases[i];

            int wr = weight.get_rows();
            int wc = weight.get_cols();

            int br = bias.get_rows();
            int bc = bias.get_cols();

            outfile.write(reinterpret_cast<char *>(&wr), sizeof(int));
            outfile.write(reinterpret_cast<char *>(&wc), sizeof(int));
            outfile.write(reinterpret_cast<char *>(weight.data()), sizeof(double) * wr * wc);

            outfile.write(reinterpret_cast<char *>(&br), sizeof(int));
            outfile.write(reinterpret_cast<char *>(&bc), sizeof(int));
            outfile.write(reinterpret_cast<char *>(bias.data()), sizeof(double) * br * bc);
        }
        outfile.close();
    }

    void Network::load(std::string filename) {
        std::ifstream infile;
        infile.open(filename, std::ios::binary | std::ios::in);
        for(int i = 0; i < _activations.size(); i++) {
            Matrix &weight = _weights[i];
            Matrix &bias = _biases[i];

            int wr, wc, br, bc;
            infile.read(reinterpret_cast<char *>(&wr), sizeof(int));
            infile.read(reinterpret_cast<char *>(&wc), sizeof(int));
            infile.read(reinterpret_cast<char *>(weight.data()), sizeof(double) * wr * wc);

            infile.read(reinterpret_cast<char *>(&br), sizeof(int));
            infile.read(reinterpret_cast<char *>(&bc), sizeof(int));
            infile.read(reinterpret_cast<char *>(bias.data()), sizeof(double) * br * bc);
        }
        infile.close();
    }
}