#include "Neural_utils.h"

double Neural_utils_max(double a, double b) {
    if(a > b) {
        return a;
    }
    else {
        return b;
    }
}

double Neural_utils_min(double a, double b) {
    if(a < b) {
        return a;
    }
    else {
        return b;
    }
}

double Neural_utils_random(void) {
    return rand()/(double)RAND_MAX;
}

int Neural_utils_randrange(int x) {
    return rand()%x;
}

void Neural_utils_swap(void *a, void *b, size_t type_size) {
    // Assumes a and b are of equal size
    void *tmp = malloc(type_size);
    memcpy(tmp, a, type_size);
    memcpy(a, b, type_size);
    memcpy(b, tmp, type_size);
    free(tmp);
}

void Neural_utils_shuffle(void *array, int length, size_t type_size) {
    for(int i = length-1; i > 0; --i) {
        int j = Neural_utils_randrange(i);
        Neural_utils_swap(
            array+(i*type_size), 
            array+(j*type_size), 
            type_size
        );
    }
}
