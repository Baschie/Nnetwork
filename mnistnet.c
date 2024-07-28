#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "nnetwork.h"
#include "reader.h"

#define TRAINIMAGES "archive/train-images.idx3-ubyte"
#define TRAINLABELS "archive/train-labels.idx1-ubyte"
#define TESTIMAGES  "archive/t10k-images.idx3-ubyte"
#define TESTLABELS  "archive/t10k-labels.idx1-ubyte"

double sigmoid(double x);
double sigmoid_derivative(double x);
double interpert_mnist(Matrix *v);

int main()
{
    srand(time(NULL));
    Dataset *trainset = read(TRAINIMAGES, TRAINLABELS, 0, 60000, 100);
    Dataset *testset = read(TESTIMAGES, TESTLABELS, 0, 10000, 10000);
    Activation activatoin = {sigmoid, sigmoid_derivative};
    Nnet *mnistnet = nnetalloc(28 * 28, (int []) {196, 10},  (Activation []) {activatoin, activatoin}, 2);
    printf("Initial accurecy: %lf%%\n", accuracy(mnistnet, testset, interpert_mnist));
    stochastic_train(mnistnet, trainset, 10, 0.05);
    printf("Final accuracy: %lf%%\n", accuracy(mnistnet, testset, interpert_mnist));
}

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));   
}

double sigmoid_derivative(double x)
{
    return x * (1 - x);
}

double interpert_mnist(Matrix *v)
{
    int maxi = 0;
    for (int i = 0; i < v->row; i++)
        if (v->entries[i] > v->entries[maxi])
            maxi = i;
    return maxi;
}