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

    memcpy(dest->entries, preout->entries, preout->row * sizeof(double));
    mtfree(preout);
    mtfree(out);
    return dest;
}

#define EPSILON 1e-8

void train(Nnet *nnet, Batch *batch, double learning_rate, Matrix *wgradients, Matrix *bgradients, Matrix *outputs)
{
    Matrix *inputs = batch->inputs, *targets = batch->targets;
    int ndata = batch->size;
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
            for (int k = 0; k < error->row; k++)
                losses[j] += error->entries[k] * error->entries[k];
            Matrix *bgrad = mtapply(&outputs[j], nnet->functions[j].fprime, mtalloc(outputs[j].row, 1));
            mtscale(bgrad, 2, bgrad);
            mtelmult(bgrad, error, bgrad);
            Matrix *itranspose = mttranspose(j > 0 ? &outputs[j - 1] : &inputs[i], mtalloc(1, nnet->weights[j].col));
            Matrix *wgrad = mtmult(bgrad, itranspose, mtalloc(bgrad->row, itranspose->col));
            mtfree(itranspose);

            if (j > 0) {
                Matrix *wtranspose = mttranspose(&nnet->weights[j], mtalloc(nnet->weights[j].col, nnet->weights[j].row));
                error->row = wtranspose->row;
                error->entries = realloc(error->entries, error->row * sizeof(double));
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
        double bnormsq = 0; /* Frobenious norm of bias gradient squared */
        for (int j = 0; j < bgradients[i].row; j++)
            bnormsq += bgradients[i].entries[j] * bgradients[i].entries[j];
        double wnormsq = 0; /* Frobenious norm of weight gradient squared */
        for (int j = 0; j < nnet->weights[i].row * nnet->weights[i].col; j++)
            wnormsq += wgradients[i].entries[j] * wgradients[i].entries[j];
        double coefficient = losses[i] / ndata / (bnormsq + wnormsq + EPSILON) * learning_rate;
        mtscale(&bgradients[i], coefficient, &bgradients[i]);
        mtsubtract(&nnet->biases[i], &bgradients[i], &nnet->biases[i]);
        memset(bgradients[i].entries, 0, bgradients[i].row * sizeof(double));
        mtscale(&wgradients[i], coefficient, &wgradients[i]);
        mtsubtract(&nnet->weights[i], &wgradients[i], &nnet->weights[i]); 
        memset(wgradients[i].entries, 0, wgradients[i].row * wgradients[i].col);
    }
}

void shuffle(int v[], int len)
{
    for (int i = 0; i < len; i++) {
        int randi = rand() % len;
        int tmp = v[i];
        v[i] = v[randi];
        v[randi] = tmp;
    }
}

void stochastic_train(Nnet *nnet, Dataset *dataset, int epoches, double learning_rate)
{
    Matrix wgradients[nnet->nlay], bgradients[nnet->nlay], outputs[nnet->nlay];

    for (int i = 0; i < nnet->nlay; i++) {
        wgradients[i].row = nnet->weights[i].row;
        wgradients[i].col = nnet->weights[i].col;
        wgradients[i].entries = calloc(wgradients[i].row * wgradients[i].col, sizeof(double));
        bgradients[i].row = nnet->biases[i].row;
        bgradients[i].col = 1;
        bgradients[i].entries = calloc(bgradients[i].row, sizeof(double));
        outputs[i].row = bgradients[i].row;
        outputs[i].col = 1;
        outputs[i].entries = calloc(outputs[i].row, sizeof(double));
    }

    int indecies[dataset->nbatch];

    for (int i = 0; i < dataset->nbatch; i++)
        indecies[i] = i;
    while (epoches-- > 0) {
        shuffle(indecies, dataset->nbatch);
        for (int i = 0; i < dataset->nbatch; i++)
            train(nnet, &dataset->batches[indecies[i]], learning_rate, wgradients, bgradients, outputs);
    }

    for (int i = 0; i < nnet->nlay; i++) {
        free(wgradients[i].entries);
        free(bgradients[i].entries);
        free(outputs[i].entries);
    }
}

double accuracy(Nnet *nnet, Dataset *dataset, double (*interpret)(Matrix *))
{
    int correct = 0;
    Matrix *prediction = mtalloc(dataset->batches->targets->row, 1);
    for (int i = 0; i < dataset->nbatch; i++)
        for (int j = 0; j < dataset->batches->size; j++) {
            predict(nnet, &dataset->batches[i].inputs[j], prediction);
            if (interpret(prediction) == interpret(&dataset->batches[i].targets[j]))
                correct++;
        }
    mtfree(prediction);
    return correct * 100.0 / dataset->nbatch / dataset->batches->size;
}