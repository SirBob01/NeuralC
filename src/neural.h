#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <errno.h>

typedef struct matrix_t {
	double *cells;
	int rows, cols;
} Matrix;

typedef struct layer_t {
	int nodes;
	double (*activation)(double, int);
} Layer;

typedef struct network_t {
	Layer *structure;
	int layers;
	double (*cost)(double, double, int);

	Matrix **states;
	Matrix **weights;
	Matrix **biases;

	Matrix **delta_w;
	Matrix **delta_b;

	int normalize;
} NeuralNetwork;

typedef struct dataset_t {
	double *inputs;
	double *expected;
} DataSet;


/* matrix.c */
Matrix *matrix(double *values, int rows, int cols);
Matrix *clone_matrix(Matrix *m);

Matrix *diagonal_matrix(double *values, int n);
Matrix *get_diagonal(Matrix *m);

void resize_matrix(Matrix *m, int rows, int cols);
void map_matrix(Matrix *m, double *values);
void copy_matrix(Matrix *target, Matrix *source);
void destroy_matrix(Matrix *m);

double get_at(Matrix *m, int row, int col);
void set_at(Matrix *m, double value, int row, int col);

void scale_ip(Matrix *m, double s);
void add_ip(Matrix *a, Matrix *b);
void sub_ip(Matrix *a, Matrix *b);
void hadamard_ip(Matrix *a, Matrix *b);
void mul_ip(Matrix *a, Matrix *b);
void transpose_ip(Matrix *m);

Matrix *scale(Matrix *m, double s);
Matrix *add(Matrix *a, Matrix *b);
Matrix *sub(Matrix *a, Matrix *b);
Matrix *hadamard(Matrix *a, Matrix *b);
Matrix *mul(Matrix *a, Matrix *b);
Matrix *transpose(Matrix *m);

int within_bounds(Matrix *m, int row, int col);
int equal_dimensions(Matrix *a, Matrix *b);
void print_matrix(Matrix *m);

/* network.c */
NeuralNetwork *neural_network(Layer *layers, int n, int normalize, double (*cost_function)(double, double, int));
Matrix *forward(NeuralNetwork *n, double *inputs);
void backward(NeuralNetwork *n, double *expected);
void train(NeuralNetwork *n, DataSet *batch, int batch_size, double learning_rate);
void destroy_network(NeuralNetwork *n);

/* activations.c */
double identity(double x, int derivative);
double l_relu(double x, int derivative);
double sigmoid(double x, int derivative);
double tan_h(double x, int derivative);
Matrix *softmax(Matrix *vector, int derivative); // Normalizing function

/* costs.c */
double quadratic(double output, double expected, int derivative);
double cross_entropy(double output, double expected, int derivative);