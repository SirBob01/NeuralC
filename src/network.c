// TODO: Implement stochastic and mini-batch learning algorithms
#include "neural.h"

NeuralNetwork *neural_network(Layer *layers, int n, int normalize, double (*cost_function)(double, double, int)) {	
	NeuralNetwork *net = malloc(sizeof(NeuralNetwork));
	if(net == NULL) {
		perror("Could not generate neural network");
	}

	net->structure = layers;
	net->cost = cost_function;
	net->layers = n;
	net->normalize = normalize;

	net->states = malloc(sizeof(Matrix *) * n);
	net->weights = malloc(sizeof(Matrix *) * (n-1));
	net->biases = malloc(sizeof(Matrix *) * (n-1));

	net->delta_w = malloc(sizeof(Matrix *) * (n-1));
	net->delta_b = malloc(sizeof(Matrix *) * (n-1));
	if(net->states == NULL || net->weights == NULL || net->biases == NULL || net->delta_w == NULL || net->delta_b == NULL) {
		perror("Could not generate neural network");
	}

	srand(time(NULL));
	for(int i = 0; i < n; ++i) {
		net->states[i] = matrix(NULL, 1, layers[i].nodes);

		if(i < n-1) {
			net->weights[i] = matrix(NULL, layers[i].nodes, layers[i+1].nodes);
			net->biases[i] = matrix(NULL, 1, layers[i+1].nodes);

			net->delta_w[i] = matrix(NULL, layers[i].nodes, layers[i+1].nodes);
			net->delta_b[i] = matrix(NULL, 1, layers[i+1].nodes);

			// Randomize weights
			for(int j = 0; j < layers[i].nodes * layers[i+1].nodes; ++j) {
				net->weights[i]->cells[j] = (double)rand()/(double)RAND_MAX;
			}
		}
	}
	return net;
}

Matrix *forward(NeuralNetwork *n, double *inputs) {
	map_matrix(n->states[0], inputs);	
	Matrix *target;
	Matrix *next = matrix(NULL, 1, 1);

	for(int i = 0; i < n->layers; ++i) {
		// a(n+1) = z(n)*w(n, n+1) + b(n+1)
		if(i < n->layers-1) {
			copy_matrix(next, n->states[i]);

			mul_ip(next, n->weights[i]);
			add_ip(next, n->biases[i]);
			
			copy_matrix(n->states[i+1], next);
			target = n->states[i+1];
		}
		else {
			target = n->states[i]; // Last layer (output)
		}

		// z(n) = f(a(n))
		// Allows custom activation functions
		for(int j = 0; j < target->rows * target->cols; ++j) {
			target->cells[j] = n->structure[i].activation(target->cells[j], 0);
		}
	}

	// Normalize output for multi-classification problems
	if(n->normalize) {
		target = softmax(n->states[n->layers-1], 0);
		copy_matrix(n->states[n->layers-1], target);
		destroy_matrix(target);
	}
	destroy_matrix(next);
	return n->states[n->layers-1];
}

void backward(NeuralNetwork *n, double *expected) {
	Matrix *exp_m = matrix(expected, 1, n->structure[n->layers-1].nodes);
	Matrix *error = matrix(NULL, 1, n->structure[n->layers-1].nodes);

	Matrix *a_delta = matrix(NULL, 1, 1);
	Matrix *p_delta = matrix(NULL, 1, 1);
	Matrix *e_delta = matrix(NULL, 1, 1);
	Matrix *w_delta = matrix(NULL, 1, 1);

	// Error vector
	for(int i = 0; i < error->rows * error->cols; ++i) {
		error->cells[i] = n->cost(n->states[n->layers-1]->cells[i], exp_m->cells[i], 1);
	}

	// Propagate backwards
	for(int i = n->layers-1; i > 0; --i) {
		copy_matrix(a_delta, n->states[i]);
		copy_matrix(p_delta, n->states[i-1]);
		copy_matrix(e_delta, error);

		if(i == n->layers-1 && n->normalize) {
			Matrix *norm = softmax(a_delta, 1);
			copy_matrix(a_delta, norm);
			destroy_matrix(norm);
		}

		for(int j = 0; j < n->structure[i].nodes; ++j) {
			a_delta->cells[j] = n->structure[i].activation(a_delta->cells[j], 1);
		}

		if(i < n->layers-1) {
			copy_matrix(w_delta, n->weights[i]);
			transpose_ip(w_delta);
			mul_ip(e_delta, w_delta);
		}

		hadamard_ip(e_delta, a_delta);
		transpose_ip(p_delta);
		mul_ip(p_delta, e_delta);

		copy_matrix(n->delta_w[i-1], p_delta);
		copy_matrix(n->delta_b[i-1], e_delta);
	}
	destroy_matrix(exp_m);
	destroy_matrix(error);

	destroy_matrix(a_delta);
	destroy_matrix(p_delta);
	destroy_matrix(e_delta);
	destroy_matrix(w_delta);
}

void train(NeuralNetwork *n, DataSet *batch, int batch_size, double learning_rate) {
	for(int i = 0; i < batch_size; ++i) {
		forward(n, batch[i].inputs);
		backward(n, batch[i].expected);

		// Update weights and biases on each pass
		for(int j = 0; j < n->layers-1; ++j) {
			scale_ip(n->delta_w[j], learning_rate/batch_size);
			sub_ip(n->weights[j], n->delta_w[j]);

			scale_ip(n->delta_b[j], learning_rate/batch_size);
			sub_ip(n->biases[j], n->delta_b[j]);
		}
	}
}

void destroy_network(NeuralNetwork *n) {
	for(int i = 0; i < n->layers; i++) {
		destroy_matrix(n->states[i]);

		if(i < n->layers-1) {
			destroy_matrix(n->weights[i]);
			destroy_matrix(n->biases[i]);

			destroy_matrix(n->delta_w[i]);
			destroy_matrix(n->delta_b[i]);
		}
	}

	free(n->states);
	free(n->weights);
	free(n->biases);

	free(n->delta_w);
	free(n->delta_b);

	free(n);
	n = NULL;
}