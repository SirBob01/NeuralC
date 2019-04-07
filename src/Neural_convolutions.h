#ifndef NEURAL_CONVOLUTIONS_H
#define NEURAL_CONVOLUTIONS_H

#include "Neural_matrix.h"

NeuralMatrix *Neural_convolve(NeuralMatrix *channel, const double **kernel);

#endif