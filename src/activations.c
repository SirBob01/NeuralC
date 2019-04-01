#include "neural.h"

double identity(double x, int derivative) {
	if(!derivative) {
		return x;
	}
	else {
		return 1.0;
	}
}

double l_relu(double x, int derivative) {
	double slope = 0.01;
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

double sigmoid(double x, int derivative) {
	double y = 1.0/(1.0+exp(x));
	if(!derivative) {
		return y;
	}
	else {
		return y * (1 - y);
	}
}

double tan_h(double x, int derivative) {
	double y = tanh(x);
	if(!derivative) {
		return y;
	}
	else {
		return 1 - pow(y, 2);
	}
}

Matrix *softmax(Matrix *vector, int derivative) {
	Matrix *n, *t, *d;
	Matrix *output = clone_matrix(vector);

	int len = output->rows * output->cols;
	double sum = 0;
	for(int i = 0; i < len; i++) {
		output->cells[i] = exp(output->cells[i]);
		sum += output->cells[i];
	}
	
	scale_ip(output, (1.0/sum));
	if(derivative) {
		n = diagonal_matrix(output->cells, len);
		t = transpose(output);
		mul_ip(t, output);
		sub_ip(n, t);

		d = get_diagonal(n);
		copy_matrix(output, d);

		destroy_matrix(n);
		destroy_matrix(t);
		destroy_matrix(d);
	}
	return output;
}