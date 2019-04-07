#ifndef NEURAL_UTILS_H
#define NEURAL_UTILS_H

#include <stdlib.h>
#include <string.h>
#include <time.h>

double Neural_max(double a, double b);

double Neural_min(double a, double b);

void Neural_swap(void *a, void *b, size_t type_size);

void Neural_shuffle(void *array, int length, size_t type_size);

#endif