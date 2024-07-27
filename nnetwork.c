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

#define EPSILON 1e-3

void train(Nnet *nnet, Matrix *inputs, Matrix *targets, int ndata, double learning_rate, Matrix *wgradients, Matrix *bgradients, Matrix *outputs)
{
    double losses[nnet->nlay];
    memset(losses, 0, sizeof(double) * nnet->nlay);

    for (int i = 0; i < ndata; i++) {
        Matrix *input = &inputs[i];

        for (int j = 0; j < nnet->nlay; j++) {
            mtmult(&nnet->weights[j], input, &outputs[j]);
            mtadd(&nnet->biases[j], &outputs[j], &outputs[j]);
            mtapply(&outputs[j], nnet->functions[j].f, &outputs[j]);
            input = &outputs[j];
        }

        Matrix *error = mtsubtract(&outputs[nnet->nlay - 1], &targets[i], mtalloc(outputs[nnet->nlay - 1].row, 1));

        for (int j = nnet->nlay - 1; j >= 0; j--) {
            Matrix *bgrad = mtapply(&outputs[j], nnet->functions[j].fprime, mtalloc(outputs[j].row, 1));
            mtscale(bgrad, 2, bgrad);
            mtelmult(bgrad, error, bgrad);
            Matrix *itranspose = mttranspose(j > 0 ? &outputs[j - 1] : &inputs[i], mtalloc(1, nnet->weights[j].col));
            Matrix *wgrad = mtmult(bgrad, itranspose, mtalloc(bgrad->row, itranspose->col));
            mtfree(itranspose);

            if (j > 0) {
                Matrix *wtranspose = mttranspose(&nnet->weights[j], mtalloc(nnet->weights[j].col, nnet->weights[j].row));
                error->row = wtranspose->row;
                error->entries = realloc(error->entries, error->row);
                mtmult(wtranspose, bgrad, error);
                mtfree(wtranspose);
            }

            mtdivide(bgrad, ndata, bgrad);
            mtadd(bgrad, &bgradients[j], &bgradients[j]);
            mtfree(bgrad);
            mtdivide(wgrad, ndata, wgrad);
            mtadd(wgrad, &wgradients[j], &wgradients[j]);
            mtfree(wgrad);
        }
        mtfree(error);
    }

    for (int i = 0; i < nnet->nlay; i++) {
        int bnormsq = 0; /* Frobenious norm of bias gradient squared */
        for (int j = 0; j < bgradients[i].row; j++)
            bnormsq += bgradients[i].entries[j] * bgradients[i].entries[j];
        int wnormsq = 0; /* Frobenious norm of weight gradient squared */
        for (int j = 0; j < nnet->weights[i].row * nnet->weights[i].col; j++)
            wnormsq += wgradients[i].entries[j] * wgradients[i].entries[j];
        double coefficient = losses[i] / (bnormsq + wnormsq + EPSILON) * learning_rate;
        mtscale(&bgradients[i], coefficient, &bgradients[i]);
        mtsubtract(&nnet->biases[i], &bgradients[i], &nnet->biases[i]);
        memset(bgradients[i].entries, 0, bgradients[i].row * sizeof(double));
        mtscale(&wgradients[i], coefficient, &wgradients[i]);
        mtsubtract(&nnet->weights[i], &wgradients[i], &nnet->weights[i]); 
        memset(wgradients[i].entries, 0, wgradients[i].row * wgradients[i].col);
    }
}