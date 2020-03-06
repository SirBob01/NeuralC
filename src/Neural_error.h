#ifndef NEURAL_ERROR_H
#define NEURAL_ERROR_H

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#define _NEURAL_ERROR_MESSAGE_BUFFER 100
#define _NEURAL_ERROR_MESSAGE_PADDING 10

typedef enum {
    SUCCESS,
    NO_NETWORK_MEMORY,
    NO_MATRIX_MEMORY,

    MATRIX_OUT_BOUNDS,
    INVALID_MATRIX_ADDITION,
    INVALID_MATRIX_SUBTRACTION,
    INVALID_MATRIX_HADAMARD,
    INVALID_MATRIX_MULTIPLICATION,

    INVALID_BATCH_SIZE,

    INVALID_ERROR_MESSAGE
} NeuralError;

typedef struct {
    int max_length;
    int current_length;
    NeuralError *log;
} NeuralErrorLog;

static NeuralErrorLog _Neural_Error_Log;
static char _Neural_Error_Message[_NEURAL_ERROR_MESSAGE_BUFFER];


void Neural_error_init(void);

void Neural_error_quit(void);

void Neural_error_message_format(const char *format, ...);

char *Neural_error_get(void);

void Neural_error_set(NeuralError error);

#endif