#include "../nnetwork.h"
#include "unity.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

double relu(double x) {
    return x > 0 ? x : 0;
}

void test_nnetalloc(void) {
    int layer_sizes[] = {3, 2, 1};
    Activation functions[] = {{relu, NULL}, {relu, NULL}, {relu, NULL}};
    int input_size = 4;
    int nlay = 3;

    Nnet *nnet = nnetalloc(input_size, layer_sizes, functions, nlay);

    TEST_ASSERT_NOT_NULL(nnet);
    TEST_ASSERT_EQUAL_INT(nlay, nnet->nlay);

    for (int i = 0; i < nlay; i++) {
        TEST_ASSERT_EQUAL_INT(layer_sizes[i], nnet->weights[i].row);
        TEST_ASSERT_EQUAL_INT(i > 0 ? layer_sizes[i - 1] : input_size, nnet->weights[i].col);
        TEST_ASSERT_EQUAL_INT(layer_sizes[i], nnet->biases[i].row);
        TEST_ASSERT_EQUAL_INT(1, nnet->biases[i].col);
    }

    nnetfree(nnet);
}

void test_predict(void) {
    int layer_sizes[] = {3, 2, 1};
    Activation functions[] = {{relu}, {relu}, {relu}};
    Matrix *input = mtalloc(4, 1);
    init(input);
    int input_size = 4;
    int nlay = 3;

    Nnet *nnet = nnetalloc(input_size, layer_sizes, functions, nlay);
    Matrix *out3 = mtmult(nnet->weights, input, mtalloc(3, 1));
    mtadd(out3, nnet->biases, out3);
    mtapply(out3, relu, out3);
    Matrix *out2 = mtmult(nnet->weights + 1, out3, mtalloc(2, 1));
    mtadd(out2, nnet->biases + 1, out2);
    mtapply(out2, relu, out2);
    Matrix *out1 = mtmult(nnet->weights + 2, out2, mtalloc(1, 1));
    mtadd(out1, nnet->biases + 2, out1);
    mtapply(out1, relu, out1);
    Matrix *result = predict(nnet, input, mtalloc(1, 1));
    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL_DOUBLE(out1->entries[0], result->entries[0]);
    mtfree(input);
    mtfree(out3);
    mtfree(out2);
    mtfree(out1);
    mtfree(result);
    nnetfree(nnet);
}



int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_nnetalloc);
    RUN_TEST(test_predict);
    return UNITY_END();
}
