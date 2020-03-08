#include "../src/Neural.h"

const double pi = 3.14159265358979323846;
const int population_size = 180;
const int batch_size = 9;

int main(int argc, char **argv) {
    Neural_init();

    // Initialize hyperparameters
    Neural_set_hyperparam_prelu(0.05);

    // Define network structure
    NeuralLayer structure[5] = {
        {1, NULL},
        {10, Neural_activation_prelu},
        {10, Neural_activation_prelu},
        {10, Neural_activation_prelu},
        {1, Neural_activation_tanh}
    };

    // Initialize network
    // softmax_output overrides output layer activation and cost functions
    NeuralNetworkDef net_def;
    net_def.structure = structure;
    net_def.layers = sizeof(structure) / sizeof(NeuralLayer),
    net_def.cost = Neural_cost_quadratic;
    net_def.softmax_output = Neural_false; 

    NeuralNetwork *net = Neural_network(net_def);
    
    // Initialize dataset
    NeuralDataPair *pairs[population_size];
    for(int j = 0; j < population_size; j++) {
        int i = j-90;
        pairs[j] = Neural_datapair(1, 1);

        // Set value map input -> expected
        pairs[j]->inputs[0] = i*pi/180;
        pairs[j]->expected[0] = sin(pairs[j]->inputs[0]);
    }
    
    // Define our test input data
    double test_data[3][1] = {{(pi/6)}, 
                              {(pi/4)}, 
                              {(pi/3)}
    };
    
    // Initial output of network
    printf("INITIAL:\n");
    for(int i = 0; i < 3; i++) {
        Neural_network_forward(net, test_data[i]);
        printf("sin(%.5f) = ", test_data[i][0]);
        Neural_matrix_print(Neural_network_output(net));
    }
    
    // Train the network 1000 times
    printf("\nTRAINING...\n");

    NeuralTrainer trainer;
    trainer.population = pairs;
    trainer.population_size = population_size;
    trainer.batch_size = batch_size;
    trainer.learning_rate = 0.01;
    
    for(int i = 0; i < 10000; ++i) {
        Neural_network_train(
            net, 
            trainer
        );

        double correct = 0;
        for(int i = 0; i < population_size; i++) {
            Neural_network_forward(net, test_data[i]);
            double error = net->def.cost(
                Neural_network_output(net)->cells[0], 
                pairs[i]->expected[0],
                Neural_false
            );
            if(error <= 0.5) {
                correct++;
            }
        }
        printf("Iteration: %d | Accuracy: %f\n", i+1, correct/population_size);
    }

    // Check output of network again
    printf("\nFINAL:\n");
    for(int j = 0; j < population_size; j++) {
        int i = j-90;
        Neural_network_forward(net, pairs[j]->inputs);
        NeuralMatrix *output = Neural_network_output(net);
        printf("sin(%.5f) = %.5f | Correct: %.5f\n", 
            pairs[j]->inputs[0],
            output->cells[0],
            pairs[j]->expected[0]
        );
    }

    // Cleanup
    for(int i = 0; i < population_size; i++) {
        Neural_datapair_destroy(pairs[i]);
    }
    Neural_network_destroy(net);
    Neural_quit();

    return 0;
}