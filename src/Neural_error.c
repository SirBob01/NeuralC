#include "Neural_error.h"

void Neural_error_init(void) {
    _Neural_Error_Log.max_length = 5;
    _Neural_Error_Log.current_length = 0;
    _Neural_Error_Log.log = malloc(
        sizeof(NeuralError) * _Neural_Error_Log.max_length
    );
    assert(_Neural_Error_Log.log); // Error logger got an error
}

void Neural_error_quit(void) {
    free(_Neural_Error_Log.log);
}

void Neural_error_message_format(const char *format, ...) {
    int len = strlen(format);
    if(len + _NEURAL_ERROR_MESSAGE_PADDING > _NEURAL_ERROR_MESSAGE_BUFFER) {
        Neural_error_set(INVALID_ERROR_MESSAGE);
    }

    va_list args;
    va_start(args, format);
    vsnprintf(
        _Neural_Error_Message, 
        len + _NEURAL_ERROR_MESSAGE_PADDING, 
        format, args
    );
}

char *Neural_error_get(void) {
    NeuralError _Neural_Error;
    if(_Neural_Error_Log.current_length == 0) {
        _Neural_Error = SUCCESS;
    }
    else {
        int last = _Neural_Error_Log.current_length-1;
        _Neural_Error = _Neural_Error_Log.log[last];
    }

    switch(_Neural_Error) {
        case SUCCESS:
            strcpy(
                _Neural_Error_Message, 
                "Successful operation!"
            );
            break;
        case INVALID_ACTIVATION_FUNCTION:
            strcpy(
                _Neural_Error_Message,
                "Invalid activation function key!"
            );
            break;
        case NO_MATRIX_MEMORY:
            strcpy(
                _Neural_Error_Message, 
                "No memory for matrix initialization."
            );
            break;
        case NO_NETWORK_MEMORY:
            strcpy(
                _Neural_Error_Message, 
                "No memory for neural network initialization."
            );
            break;
        case MATRIX_OUT_BOUNDS:
            strcpy(
                _Neural_Error_Message, 
                "Matrix access out of bounds."
            );
            break;
        case INVALID_MATRIX_ADDITION:
            strcpy(
                _Neural_Error_Message, 
                "Invalid matrix addition."
            );
            break;
        case INVALID_MATRIX_SUBTRACTION:
            strcpy(
                _Neural_Error_Message, 
                "Invalid matrix subtraction."
            );
            break;
        case INVALID_MATRIX_HADAMARD:
            strcpy(
                _Neural_Error_Message, 
                "Invalid matrix hadamard."
            );
            break;
        case INVALID_MATRIX_MULTIPLICATION:
            strcpy(
                _Neural_Error_Message, 
                "Invalid matrix multiplication."
            );
            break;
        case INVALID_BATCH_SIZE:
            strcpy(
                _Neural_Error_Message, 
                "Training population isn't divisible by batch size."
            );
            break;
        case INVALID_ERROR_MESSAGE:
            Neural_error_message_format(
                "Error message is longer than %d characters.", 
                _NEURAL_ERROR_MESSAGE_BUFFER
            );
            break;
        case -1:
            strcpy(
                _Neural_Error_Message, 
                "An unexpected error occurred."
            );
            break;
    }

    return _Neural_Error_Message;
}

void Neural_error_set(NeuralError error) {
    FILE *log_file = fopen("debug.log", "a+");

    if(_Neural_Error_Log.current_length == 0) {
        fprintf(log_file, "\n");
    }

    time_t time_current = time(NULL);
    struct tm time_format = *localtime(&time_current);

    if(_Neural_Error_Log.current_length == _Neural_Error_Log.max_length) {
        _Neural_Error_Log.max_length += 5;
        _Neural_Error_Log.log = realloc(
            _Neural_Error_Log.log, 
            sizeof(NeuralError) * _Neural_Error_Log.max_length
        );
    }

    _Neural_Error_Log.current_length++;
    _Neural_Error_Log.log[_Neural_Error_Log.current_length-1] = error;
    
    fprintf(stderr, "NeuralC encountered error code %d: %s\n", 
        error, 
        Neural_error_get()
    );
    fprintf(log_file, "ID: %d Errno.: %d -- %d-%d-%d %d:%d:%d -- %s\n", 
        _Neural_Error_Log.current_length, 
        error,
        time_format.tm_year + 1900, 
        time_format.tm_mon + 1, 
        time_format.tm_mday, 
        time_format.tm_hour, 
        time_format.tm_min, 
        time_format.tm_sec, 
        Neural_error_get()
    );
    fclose(log_file);
    exit(EXIT_FAILURE);
}