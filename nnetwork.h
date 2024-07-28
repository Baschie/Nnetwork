#pragma once
#include "matrix.h"

typedef struct {
    double (*f)(double); /* activation function */
    double (*fprime)(double); /* activation function derivative, it'll be fed f's output as its input */
} Activation;

typedef struct {
    int nlay;
    Matrix *weights;
    Matrix *biases;
    Activation *functions;
} Nnet;

typedef struct {
    int size;
    Matrix *inputs;
    Matrix *targets;
} Batch;

typedef struct {
    int nbatch;
    Batch *batches;
} Dataset;

Nnet *nnetalloc(int input_size, int *layer_sizes, Activation *functions, int nlay);
void nnetfree(Nnet *nnet);
void init(Matrix *v);
Matrix *predict(Nnet *nnet, Matrix *input, Matrix *dest);
void stochastic_train(Nnet *nnet, Dataset *dataset, int epoches, double learning_rate);
double accuracy(Nnet *nnet, Dataset *dataset, double (*interpret)(Matrix *));