#include "Neural_network.h"

NeuralNetwork *Neural_network(NeuralNetworkDef def) {    
    NeuralNetwork *net = malloc(sizeof(NeuralNetwork));
    if(!net) {
        Neural_error_set(NO_NETWORK_MEMORY);
    }

    net->layers = def.layers;
    net->cost = Neural_cost_get(def.cost_function);
    net->activations = malloc(sizeof(NeuralActivation) * net->layers);
    for(int i = 1; i < net->layers; i++) {
        net->activations[i] = Neural_activation_get(
            def.structure[i].activation_id
        );
    }

    // Allocate matrix arrays
    net->active = malloc(sizeof(NeuralMatrix *) * def.layers);
    net->input_sums = malloc(sizeof(NeuralMatrix *) * (def.layers-1));

    net->weights = malloc(sizeof(NeuralMatrix *) * (def.layers-1));
    net->biases = malloc(sizeof(NeuralMatrix *) * (def.layers-1));

    net->delta_w = malloc(sizeof(NeuralMatrix *) * (def.layers-1));
    net->delta_b = malloc(sizeof(NeuralMatrix *) * (def.layers-1));

    // Sanity check
    if(!net->activations || 
       !net->input_sums || !net->active || 
       !net->weights || !net->delta_w || 
       !net->biases || !net->delta_b) {
        Neural_error_set(NO_NETWORK_MEMORY);
    }

    for(int i = 0; i < def.layers; ++i) {
        net->active[i] = Neural_matrix(NULL, def.structure[i].nodes, 1);

        if(i < net->layers - 1) {
            net->input_sums[i] = Neural_matrix(
                NULL, def.structure[i+1].nodes, 1
            );
            net->weights[i] = Neural_matrix(
                NULL, def.structure[i+1].nodes, def.structure[i].nodes
            );
            net->biases[i] = Neural_matrix(
                NULL, def.structure[i+1].nodes, 1
            );

            net->delta_w[i] = Neural_matrix_clone(net->weights[i]);
            net->delta_b[i] = Neural_matrix_clone(net->biases[i]);

            // Randomize weights (-1, 1)
            int len_weights = net->weights[i]->rows * net->weights[i]->cols;
            for(int j = 0; j < len_weights; ++j) {
                net->weights[i]->cells[j] = (Neural_utils_random()*2)-1;
            }
        }
    }
    return net;
}

void Neural_network_destroy(NeuralNetwork *net) {
    // Leave no malloc unfreed
    for(int i = 0; i < net->layers; i++) {
        Neural_matrix_destroy(net->active[i]);

        if(i < net->layers - 1) {
            Neural_matrix_destroy(net->input_sums[i]);

            Neural_matrix_destroy(net->weights[i]);
            Neural_matrix_destroy(net->biases[i]);

            Neural_matrix_destroy(net->delta_w[i]);
            Neural_matrix_destroy(net->delta_b[i]);
        }
    }
    free(net->activations);
    free(net->active);
    free(net->input_sums);

    free(net->weights);
    free(net->biases);

    free(net->delta_w);
    free(net->delta_b);

    free(net);
}

NeuralMatrix *Neural_network_output(NeuralNetwork *net) {
    return net->active[net->layers-1];
}

void Neural_network_forward(NeuralNetwork *net, NeuralMatrix *inputs) {
    Neural_matrix_copy(net->active[0], inputs);

    // z(n) = a(n-1)*w(n, n-1) + b(n)
    // a(n) = activation(z(n))
    for(int i = 0; i < net->layers-1; ++i) {
        Neural_matrix_multiply(
            net->input_sums[i], 
            net->weights[i],
            net->active[i]
        );
        Neural_matrix_add(NULL, net->input_sums[i], net->biases[i]);

        net->activations[i+1](
            net->active[i+1],
            net->input_sums[i],
            Neural_false
        );
    }
}

void Neural_network_backward(NeuralNetwork *net, NeuralMatrix *expected) {
    NeuralMatrix *output = Neural_network_output(net);
    NeuralMatrix *act_delta = Neural_matrix(NULL, 1, 1);
    NeuralMatrix *layer_error = Neural_matrix(NULL, 1, 1);
    NeuralMatrix *output_error = Neural_matrix(NULL, 1, output->rows);
    int last_layer = net->layers-1;

    // Error gradient vector
    net->cost(output_error, output, expected, Neural_true);
    for(int i = last_layer; i > 0; i--) {
        // Calculate activation delta
        net->activations[i](
            act_delta,
            net->input_sums[i-1],
            Neural_true
        );

        // Calculate layer error
        if(i == last_layer) {
            Neural_matrix_copy(layer_error, act_delta);
            Neural_matrix_hadamard(NULL, layer_error, output_error);
        }
        else {
            NeuralMatrix *last_layer_error = net->delta_b[i];
            Neural_matrix_copy(layer_error, net->weights[i]);
            Neural_matrix_transpose(NULL, layer_error);
            Neural_matrix_multiply(
                NULL, 
                layer_error,
                last_layer_error
            );
            
            Neural_matrix_hadamard(NULL, layer_error, act_delta);
        }

        // Update the bias and weight deltas
        Neural_matrix_copy(net->delta_b[i-1], layer_error);

        Neural_matrix_transpose(NULL, layer_error);
        Neural_matrix_copy(net->delta_w[i-1], net->active[i-1]);
        Neural_matrix_multiply(NULL, net->delta_w[i-1], layer_error);
        Neural_matrix_transpose(NULL, net->delta_w[i-1]);
    }

    // Cleanup
    Neural_matrix_destroy(act_delta);
    Neural_matrix_destroy(layer_error);
    Neural_matrix_destroy(output_error);
}

void Neural_network_train(NeuralNetwork *net, NeuralTrainer trainer) {
    if(trainer.population_size % trainer.batch_size) {
        Neural_error_set(INVALID_BATCH_SIZE);
    }

    // Shuffle the population on each pass
    int order[trainer.population_size];
    for(int i = 0; i < trainer.population_size; i++) {
        order[i] = i;
    }
    Neural_utils_shuffle(
        order, 
        trainer.population_size, 
        sizeof(int)
    );

    // Average the delta for each batch and apply that to the network weights
    // batch_delta = delta_sum * learning_rate / batch_size
    double l_scale = trainer.learning_rate/((double)trainer.batch_size);

    NeuralMatrix **batch_delta_w = malloc(
        sizeof(NeuralMatrix *[net->layers-1])
    );
    NeuralMatrix **batch_delta_b = malloc(
        sizeof(NeuralMatrix *[net->layers-1])
    );

    for(int i = 0; i < net->layers-1; ++i) {
        batch_delta_w[i] = Neural_matrix_clone(net->weights[i]);
        batch_delta_b[i] = Neural_matrix_clone(net->biases[i]);
    }
    for(int i = 0; i < trainer.population_size; ++i) {
        Neural_network_forward(net, trainer.population[order[i]]->inputs);
        Neural_network_backward(net, trainer.population[order[i]]->expected);

        for(int j = 0; j < net->layers-1; ++j) {
            Neural_matrix_add(NULL, batch_delta_w[j], net->delta_w[j]);
            Neural_matrix_add(NULL, batch_delta_b[j], net->delta_b[j]);
        }

        if((i+1)%trainer.batch_size == 0) {
            // Update weights and biases on each pass
            for(int j = 0; j < net->layers-1; ++j) {
                Neural_matrix_scale(batch_delta_w[j], l_scale);
                Neural_matrix_subtract(
                    NULL, net->weights[j], batch_delta_w[j]
                );

                Neural_matrix_scale(batch_delta_b[j], l_scale);
                Neural_matrix_subtract(
                    NULL, net->biases[j], batch_delta_b[j]
                );

                // Reset values for next pass
                Neural_matrix_subtract(
                    NULL, batch_delta_w[j], batch_delta_w[j]
                );
                Neural_matrix_subtract(
                    NULL, batch_delta_b[j], batch_delta_b[j]
                );
            }
        }
    }

    // Cleanup
    for(int i = 0; i < net->layers-1; ++i) {
        Neural_matrix_destroy(batch_delta_w[i]);
        Neural_matrix_destroy(batch_delta_b[i]);
    }
    free(batch_delta_w);
    free(batch_delta_b);
}