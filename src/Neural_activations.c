#include "Neural_activations.h"


void Neural_set_hyperparam_prelu(double x) {
    HYPERPARAM_PRELU = x;
}

void Neural_set_hyperparam_elu(double x) {
    HYPERPARAM_ELU = x;
}

double Neural_activation_linear(double x, NeuralBool derivative) {
    if(!derivative) {
        return x;
    }
    else {
        return 1.0;
    }
}

double Neural_activation_relu(double x, NeuralBool derivative) {
    if(!derivative) {
        return Neural_utils_max(x, 0);
    }
    else {
        if(x > 0) {
            return 1.0;
        }
        else {
            return 0.0;
        }
    }
}

double Neural_activation_lrelu(double x, NeuralBool derivative) {
    double slope = 0.01;
    if(!derivative) {
        return Neural_utils_max(x, slope*x);
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

double Neural_activation_prelu(double x, NeuralBool derivative) {
    if(!derivative) {
        return Neural_utils_max(x, HYPERPARAM_PRELU*x);
    }
    else {
        if(x > 0) {
            return 1.0;
        }
        else {
            return HYPERPARAM_PRELU;
        }
    }
}

double Neural_activation_elu(double x, NeuralBool derivative) {
    double z = exp(x);
    if(!derivative) {
        if(x > 0) {
            return x;
        }
        else {
            return HYPERPARAM_ELU*(z - 1);
        }
    }
    else {
        if(x > 0) {
            return 1.0;
        }
        else {
            return HYPERPARAM_ELU*z;
        }
    }
}

double Neural_activation_selu(double x, NeuralBool derivative) {
    double lambda = 1.0507;
    double alpha = 1.67326;
    double y = Neural_utils_max(x, alpha*(exp(x) - 1));
    if(!derivative) {
        return lambda*y;
    }
    else {
        if(x >= 0) {
            return lambda;
        }
        else {
            return lambda*alpha*exp(x);
        }
    }
}

double Neural_activation_sigmoid(double x, NeuralBool derivative) {
    double y = 1.0/(1.0+exp(x));
    if(!derivative) {
        return y;
    }
    else {
        return y * (1 - y);
    }
}

double Neural_activation_tanh(double x, NeuralBool derivative) {
    double y = tanh(x);
    if(!derivative) {
        return y;
    }
    else {
        return 1 - pow(y, 2);
    }
}

double Neural_activation_sin(double x, NeuralBool derivative) {
    if(!derivative) {
        return sin(x);
    }
    else {
        return cos(x);
    }
}

void Neural_activation_softmax(
                    NeuralMatrix *res,
                    NeuralMatrix *vector, 
                    NeuralBool derivative) {
    if(res == NULL) {
        res = vector;
    }
    int len = vector->rows * vector->cols;
    double target[len];
    memcpy(target, vector->cells, len * sizeof(double));

    // Add a shift to each term for numerical stability
    double max = Neural_matrix_max(vector);
    double sum = 0;
    for(int i = 0; i < len; i++) {
        target[i] = exp(target[i] - max);
        sum += target[i];
    }
    for(int i = 0; i < len; i++) {
        target[i] /= sum;
    }
    
    if(derivative) {
        NeuralMatrix *jacobian = Neural_matrix_diagonal(target, len);
        for(int i = 0; i < len; i++) {
            for(int j = 0; j < len; j++) {
                double value;
                if(i == j) {
                    value = target[i] * (1 - target[i]);
                }
                else {
                    value = -target[i] * target[j];
                }
                Neural_matrix_set_at(jacobian, value, i, j);
            }
        }

        Neural_matrix_copy(res, jacobian);
        Neural_matrix_destroy(jacobian);
    }
    else {
        Neural_matrix_map(res, target);
    }
}