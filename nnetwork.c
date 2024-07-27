#include <stdlib.h>
#include <math.h>
#include "nnetwork.h"

Nnet *nnetalloc(int input_size, int *layer_sizes, Activation *functions, int nlay)
{
    Nnet *p = malloc(sizeof(Nnet));
    p->nlay = nlay;
    p->weights = malloc(sizeof(Matrix) * nlay);
    p->biases = malloc(sizeof(Matrix) * nlay);
    p->functions = functions;

    for (int i = 0; i < nlay; i++) {
        p->weights[i].row = layer_sizes[i];
        p->weights[i].col = i > 0 ? layer_sizes[i - 1] : input_size;
        p->weights[i].entries = malloc(sizeof(double) * p->weights[i].row * p->weights[i].col);
        p->biases[i].row = p->weights[i].row;
        p->biases[i].col = 1;
        p->biases[i].entries = malloc(sizeof(double) * p->biases[i].row);
        init(&p->weights[i]);
        init(&p->biases[i]);
    }

    return p;
}

void nnetfree(Nnet *nnet)
{
    for (int i = 0; i < nnet->nlay; i++) {
        free(nnet->weights[i].entries);
        free(nnet->biases[i].entries);
    }
    free(nnet->weights);
    free(nnet->biases);
    free(nnet);
}

void init(Matrix *v)
{
    double limit = sqrt(3.0 / v->col);

    for (int i = 0; i < v->row * v->col; i++)
        v->entries[i] = rand() * limit * 2 / RAND_MAX - limit;
}

Matrix *predict(Nnet *nnet, Matrix *input, Matrix *dest)
{
    Matrix *preout = mtalloc(input->row, 1);
    memcpy(preout->entries, input->entries, sizeof(double) * input->row);
    Matrix *out = mtalloc(nnet->weights[0].row, 1);

    for (int i = 0; i < nnet->nlay; i++) {
        out->row = nnet->weights[i].row;
        out->entries = realloc(out->entries, sizeof(double) * out->row);
        mtmult(nnet->weights + i, preout, out);
        mtadd(nnet->biases + i, out, out);
        mtapply(out, nnet->functions[i].f, out);
        Matrix *tmp = preout;
        preout = out;
        out = tmp;
    }

    memcpy(dest->entries, preout->entries, preout->row);
    mtfree(preout);
    mtfree(out);
    return dest;
}