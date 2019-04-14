#ifndef NEURAL_ERROR_H
#define NEURAL_ERROR_H

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define _Neural_Error_Message_Buffer 100
#define _Neural_Error_Message_Padding 10

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

NeuralErrorLog *_Neural_Error_Log;
char _Neural_Error_Message[_Neural_Error_Message_Buffer];


NeuralErrorLog *Neural_error_init(void);

void Neural_error_quit(void);

void *Neural_error_message_format(const char *format, int length, ...);

char *Neural_error_get(void);

void Neural_error_set(NeuralError error);

#endif