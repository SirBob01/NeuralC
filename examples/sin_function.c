#include "../src/Neural.h"

const double pi = 3.14159265358979323846;
const int population_size = 512;
const int batch_size = 32;

int main() {
	Neural_init();

	// Initialize hyperparameters
	HYPERPARAM_PRELU = 0.05;

	// Define network structure
	NeuralLayer layers[5] = {
		{1, Neural_activation_identity},
		{5, Neural_activation_prelu},
		{10, Neural_activation_prelu},
		{5, Neural_activation_prelu},
		{1, Neural_activation_identity}
	};

	NeuralNetwork *net = Neural_network(layers, 5, 0, Neural_cost_quadratic);
	
	// Initialize dataset
	NeuralDataSet *data = malloc(sizeof(NeuralDataSet) * population_size);
	for(int i = 0; i < population_size; i++) {
		data[i].inputs = malloc(sizeof(double));
		data[i].expected = malloc(sizeof(double));

		data[i].inputs[0] = ((i+1)*pi/180.0)/(2*pi);
		data[i].expected[0] = sin((i+1)*pi/180.0);
	}
	
	double test_data[3][1] = {{(pi/6)/(2*pi)}, 
							  {(pi/4)/(2*pi)}, 
							  {(pi/3)/(2*pi)}
	};

	printf("INITIAL:\n");
	for(int i = 0; i < 3; i++) {
		printf("sin(%.5f) = ", test_data[i][0]*(2*pi));
		Neural_matrix_print(Neural_network_forward(net, test_data[i%3]));
	}
	
	printf("\nTRAINING...\n");

	for(int i = 0; i < 1000000; ++i) {
		Neural_network_train(net, data, population_size, batch_size, 0.005);
	}

	printf("\nFINAL:\n");
	for(int i = 0; i < 3; i++) {
		printf("sin(%.5f) = ", test_data[i][0]);
		Neural_matrix_print(Neural_network_forward(net, test_data[i]));
	}

	// Don't allow memory leaks!
	Neural_network_destroy(net);

	Neural_quit();
	getchar();

	return 0;
}