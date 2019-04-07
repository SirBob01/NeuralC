#ifndef NEURAL_COSTS_H
#define NEURAL_COSTS_H

#include <math.h>

double Neural_quadratic(double output, double expected, int derivative);

double Neural_cross_entropy(double output, double expected, int derivative);

#endif