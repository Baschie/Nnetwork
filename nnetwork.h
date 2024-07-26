#pragma once
#include "matrix.h"

typedef struct {
    double (*f)(double); /* activation function */
    double (*fprime)(double); /* activation function derivative */
} Activation;

typedef struct {
    int nlay;
    Matrix *weights;
    Matrix *biases;
    Activation *functions;
} Nnet;

Nnet *nnetalloc(int input_size, int *layer_sizes, Activation *functions, int nlay);
void init(Matrix *v);