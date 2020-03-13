#include "Neural_costs.h"

void Neural_cost_init() {
    costs = hashmap_create(64, sizeof(NeuralCost));

    // Register the library implemented cost functions
    Neural_cost_register("quadratic", Neural_cost_quadratic);
    Neural_cost_register("cross_entropy", Neural_cost_cross_entropy);
}

void Neural_cost_quit() {
    hashmap_destroy(costs);
}

NeuralCost Neural_cost_get(const char *id) {
    if(id == NULL) {
        Neural_error_set(INVALID_COST_FUNCTION);
    }

    node_t *node = hashmap_get(costs, id);
    if(node == NULL) {
        Neural_error_set(INVALID_COST_FUNCTION);
    }

    return *(NeuralCost *)(node->data);
}

void Neural_cost_register(const char *id, NeuralCost func) {
    hashmap_append(costs, id, &func);
}

void Neural_cost_quadratic(
            NeuralMatrix *res,
            NeuralMatrix *output, 
            NeuralMatrix *expected, 
            NeuralBool derivative) {
    if(res == NULL) {
        res = output;
    }
    int length = res->rows * res->cols;
    for(int i = 0; i < length; i++) {
        double diff = output->cells[i] - expected->cells[i];
        if(!derivative) {
            res->cells[i] = 0.5 * diff * diff;
        }
        else {
            res->cells[i] = diff;
        }
    }
}

void Neural_cost_cross_entropy(
            NeuralMatrix *res,
            NeuralMatrix *output, 
            NeuralMatrix *expected, 
            NeuralBool derivative) {
    if(res == NULL) {
        res = output;
    }
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
