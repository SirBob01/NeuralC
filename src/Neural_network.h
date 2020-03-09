#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>
#include <time.h>

#include "Neural_matrix.h"
#include "Neural_activations.h"
#include "Neural_datapair.h"
#include "Neural_utils.h"
#include "Neural_error.h"

typedef struct {
    int nodes;
    void (*activation)(NeuralMatrix *, NeuralMatrix *, NeuralBool);
} NeuralLayer;

typedef struct {
    NeuralLayer *structure;
    int layers;

    double (*cost)(double, double, NeuralBool);
} NeuralNetworkDef;

typedef struct {
    NeuralNetworkDef def;

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

void Neural_network_forward(NeuralNetwork *net, double *inputs);

void Neural_network_backward(NeuralNetwork *net, double *expected);

void Neural_network_train(NeuralNetwork *net, NeuralTrainer trainer);

#endif