#include "../matrix.h"
#include <stdlib.h>
#include "unity.h"
#include <time.h>

void setUp(void)
{

}

void tearDown(void)
{

}

void test_mtsave_mtload(void)
{
    srand(time(NULL));
    Matrix *p = mtalloc(rand()%100, rand()%100);
    for (int i = 0; i < p->col * p->row; i++)
        p->entries[i] = rand() * 10.0 / RAND_MAX;
    int result = mtsave(p, "test.txt");
    TEST_ASSERT(result == 0);
    Matrix *loaded = mtload("test.txt");
    TEST_ASSERT(loaded != NULL);
    TEST_ASSERT_EQUAL_INT16(p->row, loaded->row);
    TEST_ASSERT_EQUAL_INT16(p->col, loaded->col);
    for (int i = 0; i < p->col * p->row; i++)
        TEST_ASSERT_EQUAL_DOUBLE(p->entries[i], loaded->entries[i]);
    mtfree(p);
    mtfree(loaded);
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_mtsave_mtload);
    return UNITY_END();
}
