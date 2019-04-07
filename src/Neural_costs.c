#include "Neural_costs.h"

double Neural_quadratic(double output, double expected, int derivative) {
	if(!derivative) {
		return 0.5 * pow(output - expected, 2);
	}
	else {
		return output - expected;
	}
}

double Neural_cross_entropy(double output, double expected, int derivative) {
	if(!derivative) {
		return -((expected * log(output)) + ((1 - expected) * log(1 - output)));
	}
	else {
		return -((expected / output) + ((1 - expected) / (1 - output)));
	}
}