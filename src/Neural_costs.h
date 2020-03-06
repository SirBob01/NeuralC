#ifndef NEURAL_COSTS_H
#define NEURAL_COSTS_H

#include <math.h>

#include "Neural_matrix.h"
#include "Neural_utils.h"

double Neural_cost_quadratic(
    double output, double expected, NeuralBool derivative
);

void Neural_cost_cross_entropy(
    NeuralMatrix *res,
    NeuralMatrix *output, 
    NeuralMatrix *expected, 
    NeuralBool derivative
);

#endif