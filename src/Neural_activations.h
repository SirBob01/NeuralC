#ifndef NEURAL_ACTIVATIONS_H
#define NEURAL_ACTIVATIONS_H

#include <math.h>

#include "Neural_matrix.h"
#include "Neural_utils.h"

// These hyperparameters must be set before using their respective functions
double HYPERPARAM_PRELU;
double HYPERPARAM_ELU;

double Neural_activation_identity(double x, NeuralBool derivative);

double Neural_activation_relu(double x, NeuralBool derivative);

double Neural_activation_lrelu(double x, NeuralBool derivative);

double Neural_activation_prelu(double x, NeuralBool derivative);

double Neural_activation_elu(double x, NeuralBool derivative);

double Neural_activation_selu(double x, NeuralBool derivative);

double Neural_activation_sigmoid(double x, NeuralBool derivative);

double Neural_activation_tanh(double x, NeuralBool derivative);

double Neural_activation_sin(double x, NeuralBool derivative);

NeuralMatrix *Neural_activation_softmax(NeuralMatrix *vector, NeuralBool derivative);

#endif