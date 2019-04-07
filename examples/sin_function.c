#include "include/neural.h"

const double pi = 3.14159265358979323846;
const int population_size = 384;
const int batch_size = 32;

int main() {
	NeuralLayer layers[4] = {
		{1, Neural_identity},
		{10, Neural_lrelu},
		{10, Neural_lrelu},
		{1, Neural_identity}
	};

	NeuralNetwork *net = Neural_network(layers, 4, 0, Neural_quadratic);
	
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
		printf("sin(%.5f) = ", test_data[i][0]);
		Neural_matrix_print(Neural_network_forward(net, test_data[i]));
	}

	printf("\nTRAINING...\n");

	for(int i = 0; i < 10000; ++i) {
		if(Neural_network_train(net, data, population_size, batch_size, 0.05) == NULL) {
			printf("\nNeural status: %s\n", Neural_get_status());
			break;
		}
	}

	printf("\nFINAL:\n");
	for(int i = 0; i < 3; i++) {
		printf("sin(%.5f) = ", test_data[i][0]);
		Neural_matrix_print(Neural_network_forward(net, test_data[i]));
	}

	Neural_network_destroy(net);

	printf("\nNeural status: %s\n", Neural_get_status());
	getchar();
	return 0;
}