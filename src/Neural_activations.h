#ifndef NEURAL_ACTIVATIONS_H
#define NEURAL_ACTIVATIONS_H

#include <math.h>

#include "Neural_matrix.h"

double Neural_identity(double x, int derivative);

double Neural_lrelu(double x, int derivative);

double Neural_sigmoid(double x, int derivative);

double Neural_tanh(double x, int derivative);

NeuralMatrix *Neural_softmax(NeuralMatrix *vector, int derivative); // Normalizing function

#endif