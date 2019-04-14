#include "Neural_network.h"

NeuralNetwork *Neural_network(NeuralLayer *layers, int n, int normalize, double (*cost_function)(double, double, int)) {	
	NeuralNetwork *net = malloc(sizeof(NeuralNetwork));
	if(!net) {
		Neural_error_set(NO_NETWORK_MEMORY);
	}

	net->structure = layers;
	net->cost = cost_function;
	net->layers = n;
	net->normalize = normalize;

	net->states = malloc(sizeof(NeuralMatrix *) * n);
	net->weights = malloc(sizeof(NeuralMatrix *) * (n-1));
	net->biases = malloc(sizeof(NeuralMatrix *) * (n-1));

	net->delta_w = malloc(sizeof(NeuralMatrix *) * (n-1));
	net->delta_b = malloc(sizeof(NeuralMatrix *) * (n-1));

	if(!(net->states && net->weights && net->biases && net->delta_w && net->delta_b)) {
		Neural_error_set(NO_NETWORK_MEMORY);
	}

	for(int i = 0; i < n; ++i) {
		net->states[i] = Neural_matrix(NULL, 1, layers[i].nodes);

		if(i < n-1) {
			net->weights[i] = Neural_matrix(NULL, layers[i].nodes, layers[i+1].nodes);
			net->biases[i] = Neural_matrix(NULL, 1, layers[i+1].nodes);

			net->delta_w[i] = Neural_matrix(NULL, layers[i].nodes, layers[i+1].nodes);
			net->delta_b[i] = Neural_matrix(NULL, 1, layers[i+1].nodes);

			// Randomize weights (-1, 1)
			for(int j = 0; j < layers[i].nodes * layers[i+1].nodes; ++j) {
				net->weights[i]->cells[j] = (Neural_utils_random()*2)-1;
			}
		}
	}
	
	return net;
}

void Neural_network_destroy(NeuralNetwork *n) {
	for(int i = 0; i < n->layers; i++) {
		Neural_matrix_destroy(n->states[i]);

		if(i < n->layers-1) {
			Neural_matrix_destroy(n->weights[i]);
			Neural_matrix_destroy(n->biases[i]);

			Neural_matrix_destroy(n->delta_w[i]);
			Neural_matrix_destroy(n->delta_b[i]);
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

NeuralMatrix *Neural_network_forward(NeuralNetwork *n, double *inputs) {
	Neural_matrix_map(n->states[0], inputs);	
	NeuralMatrix *target;
	NeuralMatrix *next = Neural_matrix(NULL, 1, 1);

	for(int i = 0; i < n->layers; ++i) {
		// a(n+1) = z(n)*w(n, n+1) + b(n+1)
		if(i < n->layers-1) {
			Neural_matrix_copy(next, n->states[i]);

			Neural_matrix_multiply(next, n->weights[i]);
			Neural_matrix_add(next, n->biases[i]);
			
			Neural_matrix_copy(n->states[i+1], next);
			target = n->states[i+1];
		}
		else {
			target = n->states[i]; // Last layer (output)
		}

		// z(n) = f(a(n))
		// Allows custom activation functions
		if(n->structure[i].activation != NULL) {
			for(int j = 0; j < target->rows * target->cols; ++j) {
				target->cells[j] = n->structure[i].activation(target->cells[j], 0);
			}
		}
	}

	if(n->normalize) {
		// Normalize output for multi-classification problems
		target = Neural_activation_softmax(n->states[n->layers-1], 0);
		Neural_matrix_copy(n->states[n->layers-1], target);
		Neural_matrix_destroy(target);
	}

	Neural_matrix_destroy(next);
	return n->states[n->layers-1];
}

void Neural_network_backward(NeuralNetwork *n, double *expected) {
	NeuralMatrix *error_gradient = Neural_matrix(NULL, 1, n->structure[n->layers-1].nodes);
	NeuralMatrix *a_delta = Neural_matrix(NULL, 1, 1);
	NeuralMatrix *p_delta = Neural_matrix(NULL, 1, 1);
	NeuralMatrix *w_delta = Neural_matrix(NULL, 1, 1);

	// Error gradient vector
	for(int i = 0; i < error_gradient->rows * error_gradient->cols; ++i) {
		error_gradient->cells[i] = n->cost(n->states[n->layers-1]->cells[i], expected[i], 1);
	}

	// Propagate backwards
	for(int i = n->layers-1; i > 0; --i) {
		Neural_matrix_copy(a_delta, n->states[i]);
		Neural_matrix_copy(p_delta, n->states[i-1]);

		if(i == n->layers-1 && n->normalize) {
			NeuralMatrix *norm = Neural_activation_softmax(a_delta, 1);
			Neural_matrix_copy(a_delta, norm);
			Neural_matrix_destroy(norm);
		}

		if(n->structure[i].activation != NULL) {
			for(int j = 0; j < n->structure[i].nodes; ++j) {
				a_delta->cells[j] = n->structure[i].activation(a_delta->cells[j], 1);
			}
		}

		if(i < n->layers-1) {
			Neural_matrix_copy(w_delta, n->weights[i]);
			Neural_matrix_transpose(w_delta);
			Neural_matrix_multiply(error_gradient, w_delta);
		}

		Neural_matrix_hadamard(error_gradient, a_delta);
		Neural_matrix_transpose(p_delta);
		Neural_matrix_multiply(p_delta, error_gradient);

		Neural_matrix_copy(n->delta_w[i-1], p_delta);
		Neural_matrix_copy(n->delta_b[i-1], error_gradient);
	}

	Neural_matrix_destroy(error_gradient);
	Neural_matrix_destroy(a_delta);
	Neural_matrix_destroy(p_delta);
	Neural_matrix_destroy(w_delta);
}

void Neural_network_train(NeuralNetwork *n, NeuralDataSet *population, int population_size, int batch_size, double learning_rate) {
	if(population_size % batch_size != 0) {
		Neural_error_set(INVALID_BATCH_SIZE);
	}

	Neural_utils_shuffle(population, population_size, sizeof(NeuralDataSet));

	NeuralMatrix **batch_delta_w = malloc(sizeof(NeuralMatrix *) * (n->layers - 1));
	NeuralMatrix **batch_delta_b = malloc(sizeof(NeuralMatrix *) * (n->layers - 1));

	double l_scale = learning_rate/((double)batch_size);

	for(int i = 0; i < n->layers-1; ++i) {
		batch_delta_w[i] = Neural_matrix(NULL, n->structure[i].nodes, n->structure[i+1].nodes);
		batch_delta_b[i] = Neural_matrix(NULL, 1, n->structure[i+1].nodes);
	}

	for(int i = 0; i < population_size; ++i) {
		Neural_network_forward(n, population[i].inputs);
		Neural_network_backward(n, population[i].expected);

		for(int j = 0; j < n->layers-1; ++j) {
			Neural_matrix_add(batch_delta_w[j], n->delta_w[j]);
			Neural_matrix_add(batch_delta_b[j], n->delta_b[j]);
		}

		if((i+1)%batch_size == 0) {
			// Update weights and biases on each pass
			for(int j = 0; j < n->layers-1; ++j) {
				Neural_matrix_scale(batch_delta_w[j], l_scale);
				Neural_matrix_subtract(n->weights[j], batch_delta_w[j]);

				Neural_matrix_scale(batch_delta_b[j], l_scale);
				Neural_matrix_subtract(n->biases[j], batch_delta_b[j]);

				// Reset values for next pass
				Neural_matrix_subtract(batch_delta_w[j], batch_delta_w[j]);
				Neural_matrix_subtract(batch_delta_b[j], batch_delta_b[j]);
			}
		}
	}

	for(int i = 0; i < n->layers-1; ++i) {
		Neural_matrix_destroy(batch_delta_w[i]);
		Neural_matrix_destroy(batch_delta_b[i]);
	}

	free(batch_delta_w);
	free(batch_delta_b);
}