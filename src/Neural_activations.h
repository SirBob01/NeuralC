#ifndef NEURAL_ACTIVATIONS_H
#define NEURAL_ACTIVATIONS_H

#include <math.h>

#include "Neural_matrix.h"
#include "Neural_utils.h"

// These hyperparameters must be set before using their respective functions
static double HYPERPARAM_PRELU;
static double HYPERPARAM_ELU;

void Neural_set_hyperparam_prelu(double x);

void Neural_set_hyperparam_elu(double x);

double Neural_activation_linear(double x, NeuralBool derivative);

double Neural_activation_relu(double x, NeuralBool derivative);

double Neural_activation_lrelu(double x, NeuralBool derivative);

double Neural_activation_prelu(double x, NeuralBool derivative);

double Neural_activation_elu(double x, NeuralBool derivative);

double Neural_activation_selu(double x, NeuralBool derivative);

double Neural_activation_sigmoid(double x, NeuralBool derivative);

double Neural_activation_tanh(double x, NeuralBool derivative);

double Neural_activation_sin(double x, NeuralBool derivative);

void Neural_activation_softmax(
    NeuralMatrix *target,
    NeuralMatrix *vector, 
    NeuralBool derivative
);

#endif