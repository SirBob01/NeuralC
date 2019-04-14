#ifndef NEURAL_ACTIVATIONS_H
#define NEURAL_ACTIVATIONS_H

#include <math.h>

#include "Neural_matrix.h"
#include "Neural_utils.h"

// These hyperparameters must be set before using their respective functions
double HYPERPARAM_PRELU;
double HYPERPARAM_ELU;

double Neural_activation_identity(double x, int derivative);

double Neural_activation_relu(double x, int derivative);

double Neural_activation_lrelu(double x, int derivative);

double Neural_activation_prelu(double x, int derivative);

double Neural_activation_elu(double x, int derivative);

double Neural_activation_selu(double x, int derivative);

double Neural_activation_sigmoid(double x, int derivative);

double Neural_activation_tanh(double x, int derivative);

double Neural_activation_sin(double x, int derivative);

NeuralMatrix *Neural_activation_softmax(NeuralMatrix *vector, int derivative); // Normalizing function

#endif