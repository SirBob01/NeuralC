#ifndef NEURAL_COSTS_H_
#define NEURAL_COSTS_H_

#include <algorithm>
#include <cmath>

namespace neural {
    inline double quadratic_cost(double predicted_y, double expected_y, bool derivative) {
        double diff = expected_y - predicted_y;
        if(derivative) {
            return diff;
        }
        else {
            return 0.5 * diff * diff;
        }
    }
}

#endif