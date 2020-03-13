#ifndef NEURAL_COSTS_H
#define NEURAL_COSTS_H

#include <math.h>

#include "Neural_matrix.h"
#include "Neural_utils.h"
#include "structures/hash.h"
#include "structures/list.h"

typedef void (*NeuralCost)(
    NeuralMatrix *, NeuralMatrix *, NeuralMatrix *, NeuralBool
);

static hashmap_t *costs;

// Database of activation functions
void Neural_cost_init();

void Neural_cost_quit();

void Neural_cost_register(const char *id, NeuralCost func);

NeuralCost Neural_cost_get(const char *id);

// Common cost functions
void Neural_cost_quadratic(
    NeuralMatrix *res,
    NeuralMatrix *output, 
    NeuralMatrix *expected, 
    NeuralBool derivative
);

void Neural_cost_cross_entropy(
    NeuralMatrix *res,
    NeuralMatrix *output, 
    NeuralMatrix *expected, 
    NeuralBool derivative
);

#endif