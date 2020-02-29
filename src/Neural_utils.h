#ifndef NEURAL_UTILS_H
#define NEURAL_UTILS_H

#include <stdlib.h>
#include <string.h>

typedef enum {
	Neural_false = 0,
	Neural_true = 1
} NeuralBool;

double Neural_utils_max(double a, double b);

double Neural_utils_min(double a, double b);

double Neural_utils_random(void);

int Neural_utils_randrange(int x);

void Neural_utils_swap(void *a, void *b, size_t type_size);

void Neural_utils_shuffle(void *array, int length, size_t type_size);

#endif