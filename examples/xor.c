#include "../src/Neural.h"

const int population_size = 4;
const int batch_size = 4;

int main(int argc, char **argv) {
    Neural_init();

    // Initialize hyperparameters
    Neural_set_hyperparam_prelu(0.05);

    // Define network structure
    NeuralLayer structure[4] = {
        {2, NULL},
        {5, "prelu"},
        {5, "prelu"},
        {1, "sigmoid"}
    };

    // Initialize network
    // softmax_output overrides output layer activation and cost functions
    NeuralNetworkDef net_def;
    net_def.structure = structure;
    net_def.layers = sizeof(structure) / sizeof(NeuralLayer);
    net_def.cost_function = "quadratic";

    NeuralNetwork *net = Neural_network(net_def);
    
    // Initialize dataset
    NeuralDataPair *pairs[population_size];

    // Define our test input data
    double test_data[4][2] = {{0.0, 0.0}, 
                              {1.0, 0.0}, 
                              {0.0, 1.0},
                              {1.0, 1.0}
    };

    for(int i = 0; i < population_size; i++) {
        pairs[i] = Neural_datapair(2, 1);
        double expected = (double)(
            (unsigned)test_data[i][0] ^ (unsigned)test_data[i][1]
        );

        // Set value map input -> expected
        Neural_matrix_map(pairs[i]->inputs, test_data[i]);
        Neural_matrix_map(pairs[i]->expected, &expected);
        printf("%.5f ^ %.5f = %.5f\n", 
            pairs[i]->inputs->cells[0], pairs[i]->inputs->cells[1],
            pairs[i]->expected->cells[0]
        );
    }
    
    // Initial output of network
    printf("INITIAL:\n");
    for(int i = 0; i < population_size; i++) {
        Neural_network_forward(net, pairs[i]->inputs);
        printf("xor(%.5f %.5f) = ", test_data[i][0], test_data[i][1]);
        Neural_matrix_print(Neural_network_output(net));
    }
    
    // Train the network 1000 times
    printf("\nTRAINING...\n");

    NeuralTrainer trainer;
    trainer.population = pairs;
    trainer.population_size = population_size;
    trainer.batch_size = batch_size;
    trainer.learning_rate = 1;

    for(int i = 0; i < 1000; ++i) {
        Neural_network_train(net, trainer);
    }

    // Check output of network again
    printf("\nFINAL:\n");
    for(int i = 0; i < population_size; i++) {
        Neural_network_forward(net, pairs[i]->inputs);
        printf("xor(%.5f %.5f) = ", test_data[i][0], test_data[i][1]);
        Neural_matrix_print(Neural_network_output(net));
    }

    // Cleanup
    for(int i = 0; i < population_size; i++) {
        Neural_datapair_destroy(pairs[i]);
    }
    Neural_network_destroy(net);
    Neural_quit();

    return 0;
}