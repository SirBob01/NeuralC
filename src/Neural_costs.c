#include "Neural_costs.h"

double Neural_cost_quadratic(double output, double expected, NeuralBool derivative) {
	if(!derivative) {
		return 0.5 * pow(output - expected, 2);
	}
	else {
		return output - expected;
	}
}

double Neural_cost_cross_entropy(double output, double expected, NeuralBool derivative) {
	if(!derivative) {
		return -((expected * log(output)) + ((1 - expected) * log(1 - output)));
	}
	else {
		return ((output - expected) / ((1 - output) * output));
	}
}
