/* 
 * TODO: 
 *   Create bitmap/png parser
 *   Implement feature parser engine for convolutional networks
 *   Read/write to file for matrices and neural networks (saving instance)
 */
#include "Neural.h"

void Neural_init(void) {
    // Initialize subsystems
    srand(time(NULL));
    Neural_error_init();
}

void Neural_quit(void) {
    Neural_error_quit();
}