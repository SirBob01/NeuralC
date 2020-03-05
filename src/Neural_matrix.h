#ifndef NEURAL_MATRIX_H
#define NEURAL_MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Neural_error.h"

typedef struct {
	double *cells;
	int rows, cols;
} NeuralMatrix;


NeuralMatrix *Neural_matrix(double *values, int rows, int cols);

NeuralMatrix *Neural_matrix_clone(NeuralMatrix *m);

NeuralMatrix *Neural_matrix_diagonal(double *values, int n);

void Neural_matrix_destroy(NeuralMatrix *m);

void Neural_matrix_copy(NeuralMatrix *target, NeuralMatrix *source);

void Neural_matrix_map(NeuralMatrix *m, double *values);

void Neural_matrix_resize(NeuralMatrix *m, int rows, int cols);

void Neural_matrix_get_diagonal(NeuralMatrix *res, NeuralMatrix *m);

double Neural_matrix_get_at(NeuralMatrix *m, int row, int col);

void Neural_matrix_set_at(NeuralMatrix *m, double value, int row, int col);

void Neural_matrix_scale(NeuralMatrix *m, double s);

void Neural_matrix_add(
	NeuralMatrix *res, NeuralMatrix *a, NeuralMatrix *b
);

void Neural_matrix_subtract(
	NeuralMatrix *res, NeuralMatrix *a, NeuralMatrix *b
);

void Neural_matrix_hadamard(
	NeuralMatrix *res, NeuralMatrix *a, NeuralMatrix *b
);

void Neural_matrix_multiply(
	NeuralMatrix *res, NeuralMatrix *a, NeuralMatrix *b
);

void Neural_matrix_transpose(NeuralMatrix *res, NeuralMatrix *m);

double Neural_matrix_sum(NeuralMatrix *m);

double Neural_matrix_min(NeuralMatrix *m);

double Neural_matrix_max(NeuralMatrix *m);

double Neural_matrix_dot(NeuralMatrix *a, NeuralMatrix *b);

short Neural_matrix_within_bounds(NeuralMatrix *m, int row, int col);

short Neural_matrix_equal_dimensions(NeuralMatrix *a, NeuralMatrix *b);

void Neural_matrix_print(NeuralMatrix *m);

#endif