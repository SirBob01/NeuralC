#include "neural.h"

Matrix *matrix(double *values, int rows, int cols) {
	Matrix *m = malloc(sizeof(Matrix));
	if(m == NULL) {
		perror("Coult not initialize matrix");
	}

	m->cells = malloc(sizeof(double) * rows * cols);
	if(m->cells == NULL) {
		perror("Could not initialize matrix");
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
		map_matrix(m, values);
	}
	return m;
}

Matrix *clone_matrix(Matrix *m) {
	Matrix *n = matrix(m->cells, m->rows, m->cols);
	return n;
}

Matrix *diagonal_matrix(double *values, int n) {
	Matrix *m = matrix(NULL, n, n);
	for(int i = 0; i < n; i++) {
		set_at(m, values[i], i, i);
	}
	return m;
}

Matrix *get_diagonal(Matrix *m) {
	Matrix *n = matrix(NULL, 1, m->cols);
	for(int i = 0; i < m->cols; i++) {
		set_at(n, get_at(m, i, i), 0, i);
	}
	return n;
}

void resize_matrix(Matrix *m, int rows, int cols) {
	if(m->rows == rows && m->cols == cols) {
		return;
	}

	m->rows = rows;
	m->cols = cols;
	m->cells = realloc(m->cells, sizeof(double) * rows * cols);
}

void map_matrix(Matrix *m, double *values) {
	memcpy(m->cells, values, sizeof(double) * m->rows * m->cols);
}

void copy_matrix(Matrix *target, Matrix *source) {
	// Does not create new Matrix object
	// Reuses memory block to prevent heap fragmentation
	resize_matrix(target, source->rows, source->cols);
	map_matrix(target, source->cells);
}

void destroy_matrix(Matrix *m) {
	free(m->cells);	
	free(m);
	m = NULL;
}

double get_at(Matrix *m, int row, int col) {
	if(!within_bounds(m, row, col)) {
		perror("Matrix indices out of bounds");
	}
	return m->cells[row * m->cols + col];
}

void set_at(Matrix *m, double value, int row, int col) {
	if(!within_bounds(m, row, col)) {
		perror("Matrix indices out of bounds");
	}
	m->cells[row * m->cols + col] = value;
}

void scale_ip(Matrix *m, double s) {
	for(int i = 0; i < m->rows * m->cols; ++i) {
		m->cells[i] *= s;
	}
}

void add_ip(Matrix *a, Matrix *b) {
	if(!equal_dimensions(a, b)) {
		perror("Invalid matrix addition");
	}
	for(int i = 0; i < a->rows * a->cols; ++i) {
		a->cells[i] += b->cells[i];
	}
}

void sub_ip(Matrix *a, Matrix *b) {
	if(!equal_dimensions(a, b)) {
		perror("Invalid matrix subtraction");
	}
	for(int i = 0; i < a->rows * a->cols; ++i) {
		a->cells[i] -= b->cells[i];
	}
}

void hadamard_ip(Matrix *a, Matrix *b) {
	if(!equal_dimensions(a, b)) {
		perror("Invalid matrix hadamard product");
	}
	for(int i = 0; i < a->rows * a->cols; ++i) {
		a->cells[i] *= b->cells[i];
	}
}

void mul_ip(Matrix *a, Matrix *b) {
	if(a->cols != b->rows) {
		perror("Invalid matrix multiplication");
	}
	// Sad naive implementation :(
	double *target = malloc(sizeof(double) * a->rows * b->cols);
	if(target == NULL) {
		perror("Matrix multiplication failed");
	}

	for(int i = 0; i < a->rows; ++i) {
		for(int j = 0; j < b->cols; ++j) {
			double dot = 0.0;
			for(int c = 0; c < a->cols; ++c) {
				dot += get_at(a, i, c) * get_at(b, c, j);
			}
			target[i * b->cols + j] = dot;
		}
	}

	resize_matrix(a, a->rows, b->cols);
	map_matrix(a, target);
	free(target);
}

void transpose_ip(Matrix *m) {
	double translate[m->rows * m->cols];
	for(int i = 0; i < m->rows * m->cols; i++) {
		int r = i/(m->cols);
		int c = i%(m->cols);
		translate[c * m->rows + r] = m->cells[i];
	}

	resize_matrix(m, m->cols, m->rows);
	map_matrix(m, translate);
}

Matrix *scale(Matrix *m, double s) {
	Matrix *n = clone_matrix(m);
	scale_ip(n, s);
	return n;
}

Matrix *add(Matrix *a, Matrix *b) {
	Matrix *m = clone_matrix(a);
	add_ip(m, b);
	return m;
}

Matrix *sub(Matrix *a, Matrix *b) {
	Matrix *m = clone_matrix(a);
	sub_ip(m, b);
	return m;
}

Matrix *hadamard(Matrix *a, Matrix *b) {
	Matrix *m = clone_matrix(a);
	add_ip(m, b);
	return m;
}

Matrix *mul(Matrix *a, Matrix *b) {
	Matrix *m = clone_matrix(a);
	mul_ip(m, b);
	return m;
}

Matrix *transpose(Matrix *m) {
	Matrix *n = clone_matrix(m);
	transpose_ip(n);
	return n;
}

int within_bounds(Matrix *m, int row, int col) {
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

int equal_dimensions(Matrix *a, Matrix *b) {
	return (a->rows == b->rows) && (a->cols == b->cols);
}

void print_matrix(Matrix *m) {
	for(int i = 0; i < m->rows; ++i) {
		for(int j = 0; j < m->cols; ++j) {
			printf("%.5f ", get_at(m, i, j));
		}
		printf("\n");
	}
}