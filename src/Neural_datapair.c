#include "Neural_datapair.h"

NeuralDataPair *Neural_datapair(int inputs, int outputs) {
    NeuralDataPair *pair = malloc(sizeof(NeuralDataPair));
    pair->inputs = Neural_matrix(NULL, inputs, 1);
    pair->expected = Neural_matrix(NULL, outputs, 1);
    return pair;
}

void Neural_datapair_destroy(NeuralDataPair *pair) {
    Neural_matrix_destroy(pair->inputs);
    Neural_matrix_destroy(pair->expected);
    free(pair);
}