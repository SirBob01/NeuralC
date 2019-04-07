#ifndef NEURAL_ERROR_H
#define NEURAL_ERROR_H

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#define __Neural_Status_Message_Length 100
#define __Neural_Status_Message_Padding 10

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

NeuralError __Neural_Status; // Default success
char __Neural_Status_Message[__Neural_Status_Message_Length];


void Neural_set_status(NeuralError status);

void Neural_format_status_message(const char *format, int length, ...);

const char *Neural_get_status(void);

#endif