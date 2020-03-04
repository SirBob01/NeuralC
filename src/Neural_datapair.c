#include "Neural_datapair.h"

NeuralDataPair *Neural_datapair(int len_inputs, int len_expected) {
	NeuralDataPair *pair = malloc(sizeof(NeuralDataPair));
	pair->len_inputs = len_inputs;
	pair->len_expected = len_expected;

	pair->inputs = calloc(pair->len_inputs, sizeof(double));
	pair->expected = calloc(pair->len_expected, sizeof(double));
	return pair;
}

void Neural_datapair_destroy(NeuralDataPair *pair) {
	free(pair->inputs);
	free(pair->expected);
	free(pair);
}