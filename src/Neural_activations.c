#include "Neural_activations.h"

void Neural_activation_init() {
    activations = hashmap_create(100, sizeof(NeuralActivation));

    // Register the library implemented activation functions
    Neural_activation_register(
        "linear",
        Neural_activation_linear
    );
    Neural_activation_register(
        "relu",
        Neural_activation_relu
    );
    Neural_activation_register(
        "lrelu",
        Neural_activation_lrelu
    );
    Neural_activation_register(
        "prelu",
        Neural_activation_prelu
    );
    Neural_activation_register(
        "elu",
        Neural_activation_elu
    );
    Neural_activation_register(
        "selu",
        Neural_activation_selu
    );
    Neural_activation_register(
        "sigmoid",
        Neural_activation_sigmoid
    );
    Neural_activation_register(
        "tanh",
        Neural_activation_tanh
    );
    Neural_activation_register(
        "sin",
        Neural_activation_sin
    );
    Neural_activation_register(
        "softmax",
        Neural_activation_softmax
    );
}

void Neural_activation_quit() {
    hashmap_destroy(activations);
}

NeuralActivation Neural_activation_get(char id[]) {
    return hashmap_get(activations, id)->data;
}

void Neural_activation_register(char id[], NeuralActivation func) {
    hashmap_append(activations, id, &func);
}

void Neural_set_hyperparam_prelu(double x) {
    HYPERPARAM_PRELU = x;
}

void Neural_set_hyperparam_elu(double x) {
    HYPERPARAM_ELU = x;
}

void Neural_activation_linear(
            NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative) {
    if(res == NULL) {
        res = m;
    }
    int width = m->rows * m->cols;
    double target[width];

    for(int i = 0; i < width; i++) {
        double x = m->cells[i];
        if(!derivative) {
            target[i] = x;
        }
        else {
            target[i] = 1.0;
        }
    }
    Neural_matrix_resize(res, m->rows, m->cols);
    Neural_matrix_map(res, target);
}

void Neural_activation_relu(
            NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative) {
    if(res == NULL) {
        res = m;
    }
    int width = m->rows * m->cols;
    double target[width];

    for(int i = 0; i < width; i++) {
        double x = m->cells[i];
        if(!derivative) {
            target[i] = Neural_utils_max(x, 0);
        }
        else {
            if(x > 0) {
                target[i] = 1.0;
            }
            else {
                target[i] = 0.0;
            }
            target[i] = 1.0;
        }
    }
    Neural_matrix_resize(res, m->rows, m->cols);
    Neural_matrix_map(res, target);
}

void Neural_activation_lrelu(
            NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative) {
    if(res == NULL) {
        res = m;
    }
    int width = m->rows * m->cols;
    double target[width];
    double slope = 0.01;

    for(int i = 0; i < width; i++) {
        double x = m->cells[i];
        if(!derivative) {
            target[i] = Neural_utils_max(x, slope*x);
        }
        else {
            if(x > 0) {
                target[i] = 1.0;
            }
            else {
                target[i] = slope;
            }
        }
    }
    Neural_matrix_resize(res, m->rows, m->cols);
    Neural_matrix_map(res, target);
}

void Neural_activation_prelu(
            NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative) {
    if(res == NULL) {
        res = m;
    }
    int width = m->rows * m->cols;
    double target[width];

    for(int i = 0; i < width; i++) {
        double x = m->cells[i];
        if(!derivative) {
            target[i] = Neural_utils_max(x, HYPERPARAM_PRELU*x);
        }
        else {
            if(x > 0) {
                target[i] = 1.0;
            }
            else {
                target[i] = HYPERPARAM_PRELU;
            }
        }
    }
    Neural_matrix_resize(res, m->rows, m->cols);
    Neural_matrix_map(res, target);
}

void Neural_activation_elu(
            NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative) {
    if(res == NULL) {
        res = m;
    }
    int width = m->rows * m->cols;
    double target[width];

    for(int i = 0; i < width; i++) {
        double x = m->cells[i];
        double z = exp(x);
        if(!derivative) {
            if(x > 0) {
                target[i] = x;
            }
            else {
                target[i] = HYPERPARAM_ELU*(z - 1);
            }
        }
        else {
            if(x > 0) {
                target[i] = 1.0;
            }
            else {
                target[i] = HYPERPARAM_ELU*z;
            }
        }
    }
    Neural_matrix_resize(res, m->rows, m->cols);
    Neural_matrix_map(res, target);
}

void Neural_activation_selu(
            NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative) {
    if(res == NULL) {
        res = m;
    }
    int width = m->rows * m->cols;
    double target[width];
    double lambda = 1.0507;
    double alpha = 1.67326;

    for(int i = 0; i < width; i++) {
        double x = m->cells[i];
        double exp_x = exp(x);
        double y = Neural_utils_max(x, alpha*(exp_x - 1));
        if(!derivative) {
            target[i] = lambda*y;
        }
        else {
            if(x >= 0) {
                target[i] = lambda;
            }
            else {
                target[i] = lambda*alpha*exp_x;
            }
        }
    }
    Neural_matrix_resize(res, m->rows, m->cols);
    Neural_matrix_map(res, target);
}

void Neural_activation_sigmoid(
            NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative) {
    if(res == NULL) {
        res = m;
    }
    int width = m->rows * m->cols;
    double target[width];

    for(int i = 0; i < width; i++) {
        double y = 1.0/(1.0+exp(m->cells[i]));
        if(!derivative) {
            target[i] = y;
        }
        else {
            target[i] = y * (1 - y);
        }
    }
    Neural_matrix_resize(res, m->rows, m->cols);
    Neural_matrix_map(res, target);
}

void Neural_activation_tanh(
            NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative) {
    if(res == NULL) {
        res = m;
    }
    int width = m->rows * m->cols;
    double target[width];

    for(int i = 0; i < width; i++) {
        double y = tanh(m->cells[i]);
        if(!derivative) {
            target[i] = y;
        }
        else {
            target[i] = 1 - pow(y, 2);
        }
    }
    Neural_matrix_resize(res, m->rows, m->cols);
    Neural_matrix_map(res, target);
}

void Neural_activation_sin(
            NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative) {
    if(res == NULL) {
        res = m;
    }
    int width = m->rows * m->cols;
    double target[width];

    for(int i = 0; i < width; i++) {
        if(!derivative) {
            target[i] = sin(m->cells[i]);
        }
        else {
            target[i] = cos(m->cells[i]);
        }
    }
    Neural_matrix_resize(res, m->rows, m->cols);
    Neural_matrix_map(res, target);
}

void Neural_activation_softmax(
            NeuralMatrix *res, NeuralMatrix *m, NeuralBool derivative) {
    if(res == NULL) {
        res = m;
    }
    int len = m->rows * m->cols;
    double target[len];
    memcpy(target, m->cells, len * sizeof(double));

    // Add a shift to each term for numerical stability
    double max = Neural_matrix_max(m);
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
        Neural_matrix_resize(res, m->rows, m->cols);
        Neural_matrix_map(res, target);
    }
}