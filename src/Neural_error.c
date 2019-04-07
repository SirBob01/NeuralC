#include "Neural_error.h"

void Neural_set_status(NeuralError status) {
	__Neural_Status = status;
}

void Neural_format_status_message(const char *format, int length, ...) {
	if(length + __Neural_Status_Message_Padding > __Neural_Status_Message_Length) {
		Neural_set_status(INVALID_ERROR_MESSAGE);
		return;
	}

	va_list args;
	va_start(args, length);
	vsnprintf(__Neural_Status_Message, length + __Neural_Status_Message_Padding, format, args);
}

const char *Neural_get_status(void) {
	switch(__Neural_Status) {
		case SUCCESS:
			strcpy(__Neural_Status_Message, "Successful operation!");
			break;
		case NO_MATRIX_MEMORY:
			strcpy(__Neural_Status_Message, "No memory for matrix initialization.");
			break;
		case NO_NETWORK_MEMORY:
			strcpy(__Neural_Status_Message, "No memory for neural network initialization.");
			break;
		case MATRIX_OUT_BOUNDS:
			strcpy(__Neural_Status_Message, "Matrix access out of bounds.");
			break;
		case INVALID_MATRIX_ADDITION:
			strcpy(__Neural_Status_Message, "Invalid matrix addition.");
			break;
		case INVALID_MATRIX_SUBTRACTION:
			strcpy(__Neural_Status_Message, "Invalid matrix subtraction.");
			break;
		case INVALID_MATRIX_HADAMARD:
			strcpy(__Neural_Status_Message, "Invalid matrix hadamard.");
			break;
		case INVALID_MATRIX_MULTIPLICATION:
			strcpy(__Neural_Status_Message, "Invalid matrix multiplication.");
			break;
		case INVALID_BATCH_SIZE:
			strcpy(__Neural_Status_Message, "Training population size isn't divisible by batch size.");
			break;
		case INVALID_ERROR_MESSAGE:
			Neural_format_status_message("Error message is longer than %d characters.", 32, __Neural_Status_Message_Length);
			break;
	}

	return __Neural_Status_Message;
}