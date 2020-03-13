#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>
#include <time.h>

#include "Neural_matrix.h"
#include "Neural_activations.h"
#include "Neural_costs.h"
#include "Neural_datapair.h"
#include "Neural_utils.h"
#include "Neural_error.h"

typedef struct {
    int nodes;
    const char *activation_id;
} NeuralLayer;

typedef struct {
    NeuralLayer *structure;
    int layers;
    const char *cost_function;
} NeuralNetworkDef;

typedef struct {
    int layers;
    NeuralCost cost;
    NeuralActivation *activations;

    NeuralMatrix **active;
    NeuralMatrix **input_sums;

    NeuralMatrix **weights;
    NeuralMatrix **biases;

    NeuralMatrix **delta_w;
    NeuralMatrix **delta_b;
} NeuralNetwork;

typedef struct {
    NeuralDataPair **population;
    
    int population_size;
    int batch_size;

    double learning_rate;
} NeuralTrainer;


NeuralNetwork *Neural_network(NeuralNetworkDef def);

void Neural_network_destroy(NeuralNetwork *net);

NeuralMatrix *Neural_network_output(NeuralNetwork *net);

void Neural_network_forward(NeuralNetwork *net, NeuralMatrix *inputs);

void Neural_network_backward(NeuralNetwork *net, NeuralMatrix *expected);

void Neural_network_train(NeuralNetwork *net, NeuralTrainer trainer);

#endif