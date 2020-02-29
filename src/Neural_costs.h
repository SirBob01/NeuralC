#ifndef NEURAL_COSTS_H
#define NEURAL_COSTS_H

#include <math.h>

#include "Neural_utils.h"

double Neural_cost_quadratic(double output, double expected, NeuralBool derivative);

double Neural_cost_cross_entropy(double output, double expected, NeuralBool derivative);

double Neural_cost_exponential(double output, double expected, NeuralBool derivative);

#endif