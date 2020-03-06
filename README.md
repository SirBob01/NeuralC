<img src="./media/logo.png" alt="NeuralC" width="500"/>

A lightweight, standalone C library for implementing deep feed-forward neural networks (DFF). It features:

- Generalized backpropagation algorithm
- Custom activation and cost functions with hyperparameters
- A full matrix library

This is simply a learning exercise hobby project.

# Dependencies

None.

# Basic Usage

1. Include the `Neural.h` header.
```c
#include <Neural.h>
```

2. Initialize the library's subsystems.
```c
Neural_init();
```

3. Create the structure of your network using an array of `NeuralLayers`, indicating the number of nodes per layer and their activation functions. Don't forget to set the hyperparameters of special activation functions like `Neural_activation_prelu`.
```c
// Initialize hyperparameters
Neural_set_hyperparam_prelu(0.05);

NeuralLayer structure[3] = {
        {1, Neural_activation_identity},
        {5, Neural_activation_prelu},
        {1, Neural_activation_identity}
};
```

4. Create a `NeuralNetworkDef` struct that defines some key properties, like the cost function. Toggling softmax normalization will override the output activation function and the cost function (cross-entropy).
```c
NeuralNetworkDef net_def;
net_def.structure = structure;
net_def.layers = sizeof(structure) / sizeof(NeuralLayer);
net_def.cost = Neural_cost_quadratic;
net_def.softmax_output = Neural_false;
```

5. Generate your neural network.
```c
NeuralNetwork *net = Neural_network(net_def);
```

6. Load your dataset into an array of `NeuralDataPair` structures {input, desired output}.
```c
int population = 100; // 100 training examples
NeuralDataPair *pair[population];
for(int i = 0; i < population; i++) {
    pair[i] = Neural_datapair(1, 1); // 1 input, 1 output

    // Set the datapair's input and expected output arrays
}
```

7. Create a `NeuralTrainer` struct that defines how the network should be trained. Note that the population must be divisible by the batch size.
```c
NeuralTrainer trainer;
trainer.population = pairs;
trainer.population_size = population;
trainer.batch_size = population / 10;
trainer.learning_rate = 0.001;
```

8. Train your network.
```c
int epochs = 1000;
for(int i = 0; i < epochs; i++) {
    Neural_network_train(net, trainer);
}
```

9. Free all allocated memory and close subsystems.
```c
// Clean-up
for(int i = 0; i < population; i++) {
    Neural_datapair_destroy(pairs[i]);
}
Neural_network_destroy(net);

// Uninitialize all subsystems
Neural_quit();
```

Read `NeuralC` source comments for more information (especially on error handling and logging). View the examples for other features.

# TODO
- Fix softmax output implementation
- Implement gradient clipping
- Read and write neural networks (and matrices) to disk
- Improve documentation