#include "Neural.h"

NeuralMatrix *Neural_matrix(double *values, int rows, int cols) {
    NeuralMatrix *m = malloc(sizeof(NeuralMatrix));
    if(!m) {
        Neural_error_set(NO_MATRIX_MEMORY);
    }

    // Default 0 matrix
    m->cells = calloc(rows * cols, sizeof(double));
    if(!m->cells) {
        Neural_error_set(NO_MATRIX_MEMORY);
    }
    m->rows = rows;
    m->cols = cols;

    if(values) {
        Neural_matrix_map(m, values);
    }
    return m;
}

NeuralMatrix *Neural_matrix_clone(NeuralMatrix *m) {
    NeuralMatrix *n = Neural_matrix(m->cells, m->rows, m->cols);
    return n;
}

NeuralMatrix *Neural_matrix_diagonal(double *values, int n) {
    NeuralMatrix *m = Neural_matrix(NULL, n, n);

    for(int i = 0; i < n; i++) {
        Neural_matrix_set_at(m, values[i], i, i);
    }

    return m;
}

void Neural_matrix_destroy(NeuralMatrix *m) {
    free(m->cells);    
    free(m);
}

void Neural_matrix_copy(NeuralMatrix *target, NeuralMatrix *source) {
    // Does not create new NeuralMatrix object
    // Reuses memory block to prevent heap fragmentation
    Neural_matrix_resize(target, source->rows, source->cols);
    Neural_matrix_map(target, source->cells);
}

void Neural_matrix_map(NeuralMatrix *m, double *values) {
    memcpy(m->cells, values, sizeof(double) * m->rows * m->cols);
}

void Neural_matrix_resize(NeuralMatrix *m, int rows, int cols) {
    if(m->rows == rows && m->cols == cols) {
        return;
    }

    m->rows = rows;
    m->cols = cols;
    m->cells = realloc(m->cells, sizeof(double) * rows * cols);
    if(!m->cells) {
        Neural_error_set(NO_MATRIX_MEMORY);
    }
}

void Neural_matrix_get_diagonal(NeuralMatrix *res, NeuralMatrix *m) {
    int len = (m->cols < m->rows) ? m->cols : m->rows;
    if(!res) {
        res = m;
    }

    double target[len];
    for(int i = 0; i < len; i++) {
        target[i] = Neural_matrix_get_at(m, i, i);
    }

    Neural_matrix_resize(res, 1, len);
    Neural_matrix_map(res, target);
}

double Neural_matrix_get_at(NeuralMatrix *m, int row, int col) {
    if(!Neural_matrix_within_bounds(m, row, col)) {
        Neural_error_set(MATRIX_OUT_BOUNDS);
    }

    return m->cells[row * m->cols + col];
}

void Neural_matrix_set_at(NeuralMatrix *m, double value, int row, int col) {
    m->cells[row * m->cols + col] = value;
}

void Neural_matrix_scale(NeuralMatrix *m, double s) {
    for(int i = 0; i < m->rows * m->cols; ++i) {
        m->cells[i] *= s;
    }
}

void Neural_matrix_add(
            NeuralMatrix *res, NeuralMatrix *a, NeuralMatrix *b) {
    if(!Neural_matrix_equal_dimensions(a, b)) {
        Neural_error_set(INVALID_MATRIX_ADDITION);
    }
    if(res == NULL) {
        res = a;
    }

    double target[a->rows * a->cols];
    for(int i = 0; i < a->rows * a->cols; ++i) {
        target[i] = a->cells[i] + b->cells[i];
    }

    Neural_matrix_resize(res, a->rows, a->cols);
    Neural_matrix_map(res, target);
}

void Neural_matrix_subtract(
            NeuralMatrix *res, NeuralMatrix *a, NeuralMatrix *b) {
    if(!Neural_matrix_equal_dimensions(a, b)) {
        Neural_error_set(INVALID_MATRIX_SUBTRACTION);
    }
    if(res == NULL) {
        res = a;
    }

    double target[a->rows * a->cols];
    for(int i = 0; i < a->rows * a->cols; ++i) {
        target[i] = a->cells[i] - b->cells[i];
    }

    Neural_matrix_resize(res, a->rows, a->cols);
    Neural_matrix_map(res, target);
}

void Neural_matrix_hadamard(
            NeuralMatrix *res, NeuralMatrix *a, NeuralMatrix *b) {
    if(!Neural_matrix_equal_dimensions(a, b)) {
        Neural_error_set(INVALID_MATRIX_HADAMARD);
    }
    if(res == NULL) {
        res = a;
    }

    double target[a->rows * a->cols];
    for(int i = 0; i < a->rows * a->cols; ++i) {
        target[i] = a->cells[i] * b->cells[i];
    }

    Neural_matrix_resize(res, a->rows, a->cols);
    Neural_matrix_map(res, target);
}

void Neural_matrix_multiply(
            NeuralMatrix *res, NeuralMatrix *a, NeuralMatrix *b) {
    if(a->cols != b->rows) {
        Neural_error_set(INVALID_MATRIX_MULTIPLICATION);
    }
    if(res == NULL) {
        res = a;
    }

    // Sad naive implementation :(
    double target[a->rows * b->cols];
    for(int i = 0; i < a->rows; ++i) {
        for(int j = 0; j < b->cols; ++j) {
            double dot = 0.0;
            for(int c = 0; c < a->cols; ++c) {
                dot += Neural_matrix_get_at(a, i, c) * 
                       Neural_matrix_get_at(b, c, j);
            }
            target[i * b->cols + j] = dot;
        }
    }
    Neural_matrix_resize(res, a->rows, b->cols);
    Neural_matrix_map(res, target);
}

void Neural_matrix_transpose(NeuralMatrix *res, NeuralMatrix *m) {
    if(res == NULL) {
        res = m;
    }

    double translate[m->rows * m->cols];
    for(int i = 0; i < m->rows * m->cols; i++) {
        int r = i/(m->cols);
        int c = i%(m->cols);
        translate[c * m->rows + r] = m->cells[i];
    }

    Neural_matrix_resize(res, m->cols, m->rows);
    Neural_matrix_map(res, translate);
}

double Neural_matrix_sum(NeuralMatrix *m) {
    double sum = 0;
    for(int i = 0; i < m->rows * m->cols; i++) {
        sum += m->cells[i];
    }

    return sum;
}

double Neural_matrix_min(NeuralMatrix *m) {
    double min = m->cells[0];
    for(int i = 0; i < m->rows * m->cols; i++) {
        if(m->cells[i] < min) {
            min = m->cells[i];
        }
    }
    return min;
}

double Neural_matrix_max(NeuralMatrix *m) {
    double max = m->cells[0];
    for(int i = 0; i < m->rows * m->cols; i++) {
        if(m->cells[i] > max) {
            max = m->cells[i];
        }
    }
    return max;
}

double Neural_matrix_dot(NeuralMatrix *a, NeuralMatrix *b) {
    NeuralMatrix *sum = Neural_matrix_clone(a);
    Neural_matrix_hadamard(NULL, sum, b);
    double dot_product = Neural_matrix_sum(sum);
    Neural_matrix_destroy(sum);
    return dot_product;
}

short Neural_matrix_within_bounds(NeuralMatrix *m, int row, int col) {
    int hor = 1;
    int ver = 1;
    if(row < 0 || row > m->rows-1) {
        hor = 0;
    }
    if(col < 0 || col > m->cols-1) {
        ver = 0;
    }

    return hor && ver;
}

short Neural_matrix_equal_dimensions(NeuralMatrix *a, NeuralMatrix *b) {
    return (a->rows == b->rows) && (a->cols == b->cols);
}

void Neural_matrix_print(NeuralMatrix *m) {
    for(int i = 0; i < m->rows; ++i) {
        for(int j = 0; j < m->cols; ++j) {
            printf("%.5f ", Neural_matrix_get_at(m, i, j));
        }
        printf("\n");
    }
}