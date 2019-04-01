#include "neural.h"

double quadratic(double output, double expected, int derivative) {
	if(!derivative) {
		return 0.5 * pow(expected - output, 2);
	}
	else {
		return expected - output;
	}
}

double cross_entropy(double output, double expected, int derivative) {
	if(!derivative) {
		return -((expected * log(output)) + ((1 - expected) * log(1 - output)));
	}
	else {
		return -((expected / output) + ((1 - expected) / (1 - output)));
	}
}