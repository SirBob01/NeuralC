#ifndef NEURAL_DATAPAIR_H
#define NEURAL_DATAPAIR_H

#include <stdlib.h>

#include "Neural_matrix.h"

typedef struct {
    NeuralMatrix *inputs;
    NeuralMatrix *expected;
} NeuralDataPair;


NeuralDataPair *Neural_datapair(int inputs, int outputs);

void Neural_datapair_destroy(NeuralDataPair *pair);

#endif