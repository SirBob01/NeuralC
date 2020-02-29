#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>
#include <time.h>

#include "Neural_matrix.h"
#include "Neural_activations.h"
#include "Neural_utils.h"
#include "Neural_error.h"

typedef struct {
	double *inputs;
	double *expected;
} NeuralDataSet;

typedef struct {
	int nodes;
	double (*activation)(double x, NeuralBool derivative);
} NeuralLayer;

typedef struct {
	NeuralLayer *structure;
	int layers;
	double (*cost)(double output, double expected, NeuralBool derivative);

	NeuralMatrix **states;
	NeuralMatrix **weights;
	NeuralMatrix **biases;

	NeuralMatrix **delta_w;
	NeuralMatrix **delta_b;

	NeuralBool normalize;
} NeuralNetwork;


NeuralNetwork *Neural_network(
	NeuralLayer *layers, int n, NeuralBool normalize, 
	double (*cost_function)(double, double, NeuralBool)
);

void Neural_network_destroy(NeuralNetwork *n);

NeuralMatrix *Neural_network_forward(NeuralNetwork *n, double *inputs);

void Neural_network_backward(NeuralNetwork *n, double *expected);

void Neural_network_train(
	NeuralNetwork *n, NeuralDataSet *population, 
	int population_size, int batch_size, double learning_rate
);

#endif