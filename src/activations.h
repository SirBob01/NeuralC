#ifndef NEURAL_ACTIVATIONS_H_
#define NEURAL_ACTIVATIONS_H_

#include <algorithm>
#include <cmath>

namespace neural {
    inline double lrelu(double x, bool derivative) {
        double slope = 0.05;
        double y = std::max(x, slope * x);
        if(derivative) {
            return y / x;
        }
        else {
            return y;
        }
    }

    inline double sigmoid(double x, bool derivative) {
        double y = 1.0/(1.0+std::exp(x));
        if(!derivative) {
            return y;
        }
        else {
            return y * (1 - y);
        }
    }

    inline double tanh(double x, bool derivative) {
        double f = 2 * sigmoid(2 * x, false) - 1;
        if(!derivative) {
            return f;
        }
        else {
            return 1 - f*f;
        }
    }
}

#endif