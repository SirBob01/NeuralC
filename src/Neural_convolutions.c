#include "Neural_convolutions.h"

const double Neural_identity_kernel[3][3] = {
	{0, 0, 0},
	{0, 1, 0},
	{0, 0, 0}
};

const double Neural_box_kernel[3][3] = {
	{1, 1, 1},
	{1, 1, 1},
	{1, 1, 1}
};

const double Neural_gauss3_kernel[3][3] = {
	{1, 2, 1},
	{2, 4, 2}, 
	{1, 2, 1}
};

const double Neural_gauss5_kernel[5][5] = {
	{1, 4, 6, 4, 1},
	{4, 16, 24, 16, 4},
	{6, 24, 36, 24, 6},
	{4, 16, 24, 16, 4},
	{1, 4, 6, 4, 1},
};

NeuralMatrix *Neural_convolve(NeuralMatrix *channel, const double **kernel) {
	NeuralMatrix *result = Neural_matrix(NULL, channel->rows, channel->cols);
	NeuralMatrix *t = Neural_matrix(NULL, sizeof(kernel[0]), sizeof(kernel[0]));
	NeuralMatrix *k = Neural_matrix((double *)kernel, sizeof(kernel[0]), sizeof(kernel[0]));

	// TODO: Implement feature parsing engine
	return result;
}