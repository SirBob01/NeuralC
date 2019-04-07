#include "Neural.h"

NeuralMatrix *Neural_matrix(double *values, int rows, int cols) {
	NeuralMatrix *m = malloc(sizeof(NeuralMatrix));
	if(m == NULL) {
		Neural_set_status(NO_MATRIX_MEMORY);
		return NULL;
	}

	m->cells = malloc(sizeof(double) * rows * cols);
	if(m->cells == NULL) {
		Neural_set_status(NO_MATRIX_MEMORY);
		return NULL;
	}

	m->rows = rows;
	m->cols = cols;

	// Default 0 matrix
	if(values == NULL) {
		for(int i = 0; i < rows * cols; ++i) {
			m->cells[i] = 0.0;
		}
	}
	else {
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
	if(m == NULL) {
		return NULL;
	}

	for(int i = 0; i < n; i++) {
		Neural_matrix_set_at(m, values[i], i, i);
	}

	return m;
}

NeuralMatrix *Neural_matrix_get_diagonal(NeuralMatrix *m) {
	NeuralMatrix *n = Neural_matrix(NULL, 1, m->cols);
	if(n == NULL) {
		return NULL;
	}

	for(int i = 0; i < m->cols; i++) {
		Neural_matrix_set_at(n, Neural_matrix_get_at(m, i, i), 0, i);
	}

	return n;
}

void *Neural_matrix_resize(NeuralMatrix *m, int rows, int cols) {
	if(m->rows == rows && m->cols == cols) {
		return NULL;
	}

	m->rows = rows;
	m->cols = cols;
	m->cells = realloc(m->cells, sizeof(double) * rows * cols);
}

void *Neural_matrix_map(NeuralMatrix *m, double *values) {
	memcpy(m->cells, values, sizeof(double) * m->rows * m->cols);
}

void *Neural_matrix_copy(NeuralMatrix *target, NeuralMatrix *source) {
	// Does not create new NeuralMatrix object
	// Reuses memory block to prevent heap fragmentation
	Neural_matrix_resize(target, source->rows, source->cols);
	Neural_matrix_map(target, source->cells);
}

void *Neural_matrix_destroy(NeuralMatrix *m) {
	free(m->cells);	
	free(m);
	m = NULL;
}

double Neural_matrix_get_at(NeuralMatrix *m, int row, int col) {
	if(!Neural_matrix_within_bounds(m, row, col)) {
		Neural_set_status(MATRIX_OUT_BOUNDS);
	}

	return m->cells[row * m->cols + col];
}

void *Neural_matrix_set_at(NeuralMatrix *m, double value, int row, int col) {
	if(!Neural_matrix_within_bounds(m, row, col)) {
		Neural_set_status(MATRIX_OUT_BOUNDS);
		return NULL;
	}

	m->cells[row * m->cols + col] = value;
}

void *Neural_matrix_scale(NeuralMatrix *m, double s) {
	for(int i = 0; i < m->rows * m->cols; ++i) {
		m->cells[i] *= s;
	}
}

void *Neural_matrix_add(NeuralMatrix *a, NeuralMatrix *b) {
	if(!Neural_matrix_equal_dimensions(a, b)) {
		Neural_set_status(INVALID_MATRIX_ADDITION);
		return NULL;
	}

	for(int i = 0; i < a->rows * a->cols; ++i) {
		a->cells[i] += b->cells[i];
	}
}

void *Neural_matrix_subtract(NeuralMatrix *a, NeuralMatrix *b) {
	if(!Neural_matrix_equal_dimensions(a, b)) {
		Neural_set_status(INVALID_MATRIX_SUBTRACTION);
		return NULL;
	}

	for(int i = 0; i < a->rows * a->cols; ++i) {
		a->cells[i] -= b->cells[i];
	}
}

void *Neural_matrix_hadamard(NeuralMatrix *a, NeuralMatrix *b) {
	if(!Neural_matrix_equal_dimensions(a, b)) {
		Neural_set_status(INVALID_MATRIX_HADAMARD);
		return NULL;
	}

	for(int i = 0; i < a->rows * a->cols; ++i) {
		a->cells[i] *= b->cells[i];
	}
}

void *Neural_matrix_multiply(NeuralMatrix *a, NeuralMatrix *b) {
	if(a->cols != b->rows) {
		Neural_set_status(INVALID_MATRIX_MULTIPLICATION);
		return NULL;
	}

	// Sad naive implementation :(
	double *target = malloc(sizeof(double) * a->rows * b->cols);
	if(target == NULL) {
		Neural_set_status(NO_MATRIX_MEMORY);
		return NULL;
	}

	for(int i = 0; i < a->rows; ++i) {
		for(int j = 0; j < b->cols; ++j) {
			double dot = 0.0;
			for(int c = 0; c < a->cols; ++c) {
				dot += Neural_matrix_get_at(a, i, c) * Neural_matrix_get_at(b, c, j);
			}
			target[i * b->cols + j] = dot;
		}
	}

	Neural_matrix_resize(a, a->rows, b->cols);
	Neural_matrix_map(a, target);
	free(target);
}

void Neural_matrix_transpose(NeuralMatrix *m) {
	double translate[m->rows * m->cols];
	for(int i = 0; i < m->rows * m->cols; i++) {
		int r = i/(m->cols);
		int c = i%(m->cols);
		translate[c * m->rows + r] = m->cells[i];
	}

	Neural_matrix_resize(m, m->cols, m->rows);
	Neural_matrix_map(m, translate);
}

double Neural_matrix_element_sum(NeuralMatrix *m) {
	double sum = 0;
	for(int i = 0; i < m->rows * m->cols; i++) {
		sum += m->cells[i];
	}

	return sum;
}

int Neural_matrix_within_bounds(NeuralMatrix *m, int row, int col) {
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

int Neural_matrix_equal_dimensions(NeuralMatrix *a, NeuralMatrix *b) {
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