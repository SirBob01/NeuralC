#ifndef NEURAL_H
#define NEURAL_H

#include <stdlib.h>
#include <time.h>

#include "Neural_network.h"
#include "Neural_datapair.h"
#include "Neural_activations.h"
#include "Neural_costs.h"
#include "Neural_matrix.h"
#include "Neural_utils.h"
#include "Neural_error.h"
#include "structures/list.h"
#include "structures/hash.h"

void Neural_init(void);

void Neural_quit(void);

#endif