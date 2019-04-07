#include "Neural_activations.h"

double Neural_identity(double x, int derivative) {
	if(!derivative) {
		return x;
	}
	else {
		return 1.0;
	}
}

double Neural_lrelu(double x, int derivative) {
	double slope = 0.03;
	if(!derivative) {
		if(x > 0) {
			return x;
		}
		else {
			return slope * x;
		}
	}
	else {
		if(x > 0) {
			return 1.0;
		}
		else {
			return slope;
		}
	}
}

double Neural_sigmoid(double x, int derivative) {
	double y = 1.0/(1.0+exp(x));
	if(!derivative) {
		return y;
	}
	else {
		return y * (1 - y);
	}
}

double Neural_tanh(double x, int derivative) {
	double y = tanh(x);
	if(!derivative) {
		return y;
	}
	else {
		return 1 - pow(y, 2);
	}
}

NeuralMatrix *Neural_softmax(NeuralMatrix *vector, int derivative) {
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