#include "include/neural.h"

const double pi = 3.14159265358979323846;
const int batch_size = 360;

int main() {
	Layer layers[3] = {
		{1, sigmoid},
		{3, l_relu},
		{1, sigmoid}
	};
	NeuralNetwork *net = neural_network(layers, 3, 1, cross_entropy);

	DataSet *data = malloc(sizeof(DataSet) * batch_size);
	for(int i = 0; i < batch_size; i++) {
		data[i].inputs = malloc(sizeof(double));
		data[i].expected = malloc(sizeof(double));

		data[i].inputs[0] = (i*pi/180.0)/(2*pi);
		data[i].expected[0] = sin(i*pi/180.0);
	}

	double test_input[1] = {(pi/6)/(2*pi)};

	printf("INITIAL:\n");
	print_matrix(forward(net, test_input));

	printf("TRAINING...\n");
	for(int i = 0; i < 1000; ++i) {
		train(net, data, batch_size, 0.01);
	}
	
	printf("FINAL:\n");
	print_matrix(forward(net, test_input));

	destroy_network(net);

	return 0;
}