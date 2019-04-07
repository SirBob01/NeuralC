#include "Neural_utils.h"

double Neural_max(double a, double b) {
	if(a >= b) {
		return a;
	}
	else {
		return b;
	}
}

double Neural_min(double a, double b) {
	if(a <= b) {
		return a;
	}
	else {
		return b;
	}
}

void Neural_swap(void *a, void *b, size_t type_size) {
	// Assumes a and b are of equal size
	void *tmp = malloc(type_size);
	memcpy(tmp, a, type_size);
	memcpy(a, b, type_size);
	memcpy(b, tmp, type_size);
}

void Neural_shuffle(void *array, int length, size_t type_size) {
	srand(time(NULL));
	for(int i = length-1; i > 0; --i) {
		int j = rand()%i;
		Neural_swap(array+(i*type_size), array+(j*type_size), type_size);
	}
}
