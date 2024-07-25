#include "../matrix.h"
#include <stdlib.h>
#include "unity.h"
#include <time.h>

void setUp(void)
{

}

void tearDown(void)
{
    mterrno = 0;
}

void test_mtalloc_failure(void)
{
    TEST_ASSERT_NULL(mtalloc(-1, 1));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_ALLOC, mterrno);
}

void test_mtfree_failure(void)
{
    mtfree(NULL);
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_NULL_MATRIX, mterrno);
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

void test_mtadd_valid(void)
{
    Matrix *p = FILL(mtalloc(2, 2), 1, 2, 3, 4);
    Matrix *q = FILL(mtalloc(2, 2), 5, 6, 7, 8);
    Matrix *dest = mtalloc(2, 2);

    mtadd(p, q, dest);

    TEST_ASSERT_EQUAL_DOUBLE(6.0, dest->entries[0]);
    TEST_ASSERT_EQUAL_DOUBLE(8.0, dest->entries[1]);
    TEST_ASSERT_EQUAL_DOUBLE(10.0, dest->entries[2]);
    TEST_ASSERT_EQUAL_DOUBLE(12.0, dest->entries[3]);

    mtfree(p);
    mtfree(q);
    mtfree(dest);
}

void test_mtsubtract_valid(void)
{
    Matrix *p = FILL(mtalloc(2, 2), 1.0, 2.0, 3.0, 4.0);
    Matrix *q = FILL(mtalloc(2, 2), 5.0, 6.0, 7.0, 7.0);
    Matrix *dest = mtalloc(2, 2);

    mtsubtract(p, q, dest);

    TEST_ASSERT_EQUAL_DOUBLE(-4.0, dest->entries[0]);
    TEST_ASSERT_EQUAL_DOUBLE(-4.0, dest->entries[1]);
    TEST_ASSERT_EQUAL_DOUBLE(-4.0, dest->entries[2]);
    TEST_ASSERT_EQUAL_DOUBLE(-3.0, dest->entries[3]);

    mtfree(p);
    mtfree(q);
    mtfree(dest);
}

void test_mtelmult_valid(void)
{
    Matrix *p = FILL(mtalloc(2, 2), 1, 2, 3, 4);
    Matrix *q = FILL(mtalloc(2, 2), 5, 6, 7, 8);
    Matrix *dest = mtalloc(2, 2);

    mtelmult(p, q, dest);

    TEST_ASSERT_EQUAL_DOUBLE(5.0, dest->entries[0]);
    TEST_ASSERT_EQUAL_DOUBLE(12.0, dest->entries[1]);
    TEST_ASSERT_EQUAL_DOUBLE(21.0, dest->entries[2]);
    TEST_ASSERT_EQUAL_DOUBLE(32.0, dest->entries[3]);

    mtfree(p);
    mtfree(q);
    mtfree(dest);
}

void test_mtadd_intcompatible(void) 
{
    Matrix *p = mtalloc(1, 2);
    Matrix *q = mtalloc(2, 2);
    Matrix *dest = mtalloc(2, 4);

    TEST_ASSERT_NULL(mtadd(p, q, dest));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_INCOMPATIBLE, mterrno);
    mterrno = 0;
    mtfree(p);
    mtfree(q);
    p = mtalloc(2, 2);
    q = mtalloc(2, 1);
    TEST_ASSERT_NULL(mtadd(p, q, dest));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_INCOMPATIBLE, mterrno);
    mterrno = 0;
    mtfree(q);
    q = mtalloc(2, 2);
    TEST_ASSERT_NULL(mtadd(p, q, dest));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_INCOMPATIBLE, mterrno);

    mtfree(p);
    mtfree(q);
    mtfree(dest);
}

void test_mtsubtract_incompatible(void) 
{
    Matrix *p = mtalloc(1, 2);
    Matrix *q = mtalloc(2, 2);
    Matrix *dest = mtalloc(2, 4);

    TEST_ASSERT_NULL(mtsubtract(p, q, dest));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_INCOMPATIBLE, mterrno);
    mterrno = 0;
    mtfree(p);
    mtfree(q);
    p = mtalloc(2, 2);
    q = mtalloc(2, 1);
    TEST_ASSERT_NULL(mtsubtract(p, q, dest));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_INCOMPATIBLE, mterrno);
    mterrno = 0;
    mtfree(q);
    q = mtalloc(2, 2);
    TEST_ASSERT_NULL(mtsubtract(p, q, dest));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_INCOMPATIBLE, mterrno);

    mtfree(p);
    mtfree(q);
    mtfree(dest);
}

void test_mtelmult_incompatible(void) {
    Matrix *p = mtalloc(1, 2);
    Matrix *q = mtalloc(2, 2);
    Matrix *dest = mtalloc(2, 4);

    TEST_ASSERT_NULL(mtelmult(p, q, dest));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_INCOMPATIBLE, mterrno);
    mterrno = 0;
    mtfree(p);
    mtfree(q);
    p = mtalloc(2, 2);
    q = mtalloc(2, 1);
    TEST_ASSERT_NULL(mtelmult(p, q, dest));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_INCOMPATIBLE, mterrno);
    mterrno = 0;
    mtfree(q);
    q = mtalloc(2, 2);
    TEST_ASSERT_NULL(mtelmult(p, q, dest));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_INCOMPATIBLE, mterrno);

    mtfree(p);
    mtfree(q);
    mtfree(dest);
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_mtalloc_failure);
    RUN_TEST(test_mtfree_failure);
    RUN_TEST(test_mtsave_mtload);
    RUN_TEST(test_mtadd_valid);
    RUN_TEST(test_mtsubtract_valid);
    RUN_TEST(test_mtelmult_valid);
    RUN_TEST(test_mtadd_intcompatible);
    RUN_TEST(test_mtsubtract_incompatible);
    RUN_TEST(test_mtelmult_incompatible);
    return UNITY_END();
}
