#include "Neural_costs.h"

double Neural_cost_quadratic(
            double output, double expected, NeuralBool derivative) {
    double diff = output - expected;
    if(!derivative) {
        return 0.5 * diff * diff;
    }
    else {
        return diff;
    }
}

void Neural_cost_cross_entropy(
            NeuralMatrix *res,
            NeuralMatrix *output, 
            NeuralMatrix *expected, 
            NeuralBool derivative) {
    Neural_matrix_copy(res, expected);
    int length = res->rows * res->cols;
    if(!derivative) {
        int cost = 0;
        for(int i = 0; i < length; i++) {
            cost += expected->cells[i] * log(output->cells[i]);
        }
        for(int i = 0; i < length; i++) {
            res->cells[i] = -cost;
        }
    }
    else {
        for(int i = 0; i < length; i++) {
            res->cells[i] /= output->cells[i];
        }
        Neural_matrix_scale(res, -1);
    }
}
