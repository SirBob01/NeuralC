#ifndef NEURAL_DATAPAIR_H
#define NEURAL_DATAPAIR_H

#include <stdlib.h>

typedef struct {
	double *inputs;
	double *expected;

	int len_inputs;
	int len_expected;
} NeuralDataPair;


NeuralDataPair *Neural_datapair(int inputs, int outputs);

void Neural_datapair_destroy(NeuralDataPair *pair);

#endif