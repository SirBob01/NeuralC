#include "Neural_activations.h"

double Neural_activation_identity(double x, int derivative) {
	if(!derivative) {
		return x;
	}
	else {
		return 1.0;
	}
}

double Neural_activation_relu(double x, int derivative) {
	double y = Neural_utils_max(x, 0);
	if(!derivative) {
		return y;
	}
	else {
		return y/x;
	}
}

double Neural_activation_lrelu(double x, int derivative) {
	double slope = 0.01;
	double y = Neural_utils_max(x, 0.01*x);
	if(!derivative) {
		return y;
	}
	else {
		return y/x;
	}
}

double Neural_activation_prelu(double x, int derivative) {
	double y = Neural_utils_max(x, HYPERPARAM_PRELU*x);
	if(!derivative) {
		return y;
	}
	else {
		return y/x;
	}
}

double Neural_activation_elu(double x, int derivative) {
	double y = Neural_utils_max(x, HYPERPARAM_ELU*(exp(x) - 1));
	if(!derivative) {
		return y;
	}
	else {
		if(x >= 0) {
			return 1.0;
		}
		else {
			return HYPERPARAM_ELU*exp(x);
		}
	}
}

double Neural_activation_selu(double x, int derivative) {
	double lambda = 1.0507;
	double alpha = 1.67326;
	double y = Neural_utils_max(x, alpha*(exp(x) - 1));
	if(!derivative) {
		return lambda*y;
	}
	else {
		if(x >= 0) {
			return lambda;
		}
		else {
			return lambda*alpha*exp(x);
		}
	}
}

double Neural_activation_sigmoid(double x, int derivative) {
	double y = 1.0/(1.0+exp(x));
	if(!derivative) {
		return y;
	}
	else {
		return y * (1 - y);
	}
}

double Neural_activation_tanh(double x, int derivative) {
	double y = tanh(x);
	if(!derivative) {
		return y;
	}
	else {
		return 1 - pow(y, 2);
	}
}

double Neural_activation_sin(double x, int derivative) {
	if(!derivative) {
		return sin(x);
	}
	else {
		return cos(x);
	}
}

NeuralMatrix *Neural_activation_softmax(NeuralMatrix *vector, int derivative) {
	NeuralMatrix *n, *t, *d;
	NeuralMatrix *output = Neural_matrix_clone(vector);

	int len = output->rows * output->cols;
	double sum = 0;
	for(int i = 0; i < len; i++) {
		output->cells[i] = exp(output->cells[i]);
		sum += output->cells[i];
	}
	
	Neural_matrix_scale(output, (1.0/sum));
	if(derivative) {
		n = Neural_matrix_diagonal(output->cells, len);
		t = Neural_matrix_clone(output);

		Neural_matrix_transpose(t);
		Neural_matrix_multiply(t, output);
		Neural_matrix_subtract(n, t);

		d = Neural_matrix_get_diagonal(n);
		Neural_matrix_copy(output, d);

		Neural_matrix_destroy(n);
		Neural_matrix_destroy(t);
		Neural_matrix_destroy(d);
	}

	return output;
}