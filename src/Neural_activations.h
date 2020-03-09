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

void Neural_activation_linear(
    NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative
);

void Neural_activation_relu(
    NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative
);

void Neural_activation_lrelu(
    NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative
);

void Neural_activation_prelu(
    NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative
);

void Neural_activation_elu(
    NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative
);

void Neural_activation_selu(
    NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative
);

void Neural_activation_sigmoid(
    NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative
);

void Neural_activation_tanh(
    NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative
);

void Neural_activation_sin(
    NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative
);

void Neural_activation_softmax(
    NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative
);

#endif