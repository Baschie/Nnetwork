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
    FILE *fp = fopen("test.txt", "wb");
    int result = mtsave(p, fp);
    fclose(fp);
    TEST_ASSERT(result == 0);
    fp = fopen("test.txt", "rb");
    Matrix *loaded = mtload(&(Matrix) {}, fp);
    fclose(fp);
    TEST_ASSERT(loaded != NULL);
    TEST_ASSERT_EQUAL_INT(p->row, loaded->row);
    TEST_ASSERT_EQUAL_INT(p->col, loaded->col);
    for (int i = 0; i < p->col * p->row; i++)
        TEST_ASSERT_EQUAL_DOUBLE(p->entries[i], loaded->entries[i]);
    mtfree(p);
    free(loaded->entries);
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

void test_mtscale_valid(void)
{
    Matrix *p = FILL(mtalloc(2, 2), 1.0, 2.0, 3.0, 4.0);
    Matrix *dest = mtalloc(2, 2);

    mtscale(p, 2.0, dest);

    TEST_ASSERT_EQUAL_DOUBLE(2.0, dest->entries[0]);
    TEST_ASSERT_EQUAL_DOUBLE(4.0, dest->entries[1]);
    TEST_ASSERT_EQUAL_DOUBLE(6.0, dest->entries[2]);
    TEST_ASSERT_EQUAL_DOUBLE(8.0, dest->entries[3]);

    mtfree(p);
    mtfree(dest);
}

void test_mtscale_incompatible(void)
{
    Matrix *p = mtalloc(2, 2);
    Matrix *dest = mtalloc(3, 3);

    TEST_ASSERT_NULL(mtscale(p, 2.0, dest));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_INCOMPATIBLE, mterrno);

    mtfree(p);
    mtfree(dest);
}

void test_mtdivide_valid(void)
{
    Matrix *p = FILL(mtalloc(2, 2), 2.0, 4.0, 6.0, 8.0);
    Matrix *dest = mtalloc(2, 2);

    mtdivide(p, 2.0, dest);

    TEST_ASSERT_EQUAL_DOUBLE(1.0, dest->entries[0]);
    TEST_ASSERT_EQUAL_DOUBLE(2.0, dest->entries[1]);
    TEST_ASSERT_EQUAL_DOUBLE(3.0, dest->entries[2]);
    TEST_ASSERT_EQUAL_DOUBLE(4.0, dest->entries[3]);

    mtfree(p);
    mtfree(dest);
}

void test_mtdivide_incompatible(void)
{
    Matrix *p = mtalloc(2, 2);
    Matrix *dest = mtalloc(3, 3);

    TEST_ASSERT_NULL(mtdivide(p, 2.0, dest));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_INCOMPATIBLE, mterrno);

    mtfree(p);
    mtfree(dest);
}

double example_func(double x) {
    return x * x;
}

void test_mtapply_valid(void)
{
    Matrix *p = FILL(mtalloc(2, 2), 1.0, 2.0, 3.0, 4.0);
    Matrix *dest = mtalloc(2, 2);

    mtapply(p, example_func, dest);

    TEST_ASSERT_EQUAL_DOUBLE(1.0, dest->entries[0]);
    TEST_ASSERT_EQUAL_DOUBLE(4.0, dest->entries[1]);
    TEST_ASSERT_EQUAL_DOUBLE(9.0, dest->entries[2]);
    TEST_ASSERT_EQUAL_DOUBLE(16.0, dest->entries[3]);

    mtfree(p);
    mtfree(dest);
}

void test_mtapply_incompatible(void)
{
    Matrix *p = mtalloc(2, 2);
    Matrix *dest = mtalloc(3, 3);

    TEST_ASSERT_NULL(mtapply(p, example_func, dest));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_INCOMPATIBLE, mterrno);

    mtfree(p);
    mtfree(dest);
}

void test_mtmult_valid(void)
{
    Matrix *p = FILL(mtalloc(2, 3), 1, 2, 3, 4, 5, 6);
    Matrix *q = FILL(mtalloc(3, 2), 7, 8, 9, 10, 11, 12);
    Matrix *dest = mtalloc(2, 2);

    mtmult(p, q, dest);

    TEST_ASSERT_EQUAL_DOUBLE(58.0, dest->entries[0]);
    TEST_ASSERT_EQUAL_DOUBLE(64.0, dest->entries[1]);
    TEST_ASSERT_EQUAL_DOUBLE(139.0, dest->entries[2]);
    TEST_ASSERT_EQUAL_DOUBLE(154.0, dest->entries[3]);
    TEST_ASSERT_EQUAL_UINT32(0, mterrno);

    mtfree(p);
    mtfree(q);
    mtfree(dest);
}

void test_mtmult_incompatible_matrices(void)
{
    Matrix *p = mtalloc(2, 2);
    Matrix *q = mtalloc(3, 3);
    Matrix *dest = mtalloc(2, 3);

    TEST_ASSERT_NULL(mtmult(p, q, dest));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_INCOMPATIBLE, mterrno);

    mtfree(p);
    mtfree(q);
    mtfree(dest);
}

void test_mtmult_incompatible_dest(void)
{
    Matrix *p = mtalloc(2, 2);
    Matrix *q = mtalloc(2, 2);
    Matrix *dest = mtalloc(3, 3);

    TEST_ASSERT_NULL(mtmult(p, q, dest));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_INCOMPATIBLE, mterrno);

    mtfree(p);
    mtfree(q);
    mtfree(dest);
}

void test_mtmult_null(void)
{
    Matrix *p = mtalloc(2, 2);
    Matrix *q = mtalloc(2, 2);
    Matrix *dest = mtalloc(2, 2);

    TEST_ASSERT_NULL(mtmult(NULL, q, dest));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_NULL_MATRIX, mterrno);
    mterrno = 0;
    TEST_ASSERT_NULL(mtmult(p, NULL, dest));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_NULL_MATRIX, mterrno);
    mterrno = 0;
    TEST_ASSERT_NULL(mtmult(p, q, NULL));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_NULL_MATRIX, mterrno);
    mterrno = 0;
    TEST_ASSERT_NULL(mtmult(NULL, NULL, NULL));
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_NULL_MATRIX, mterrno);
    
    mtfree(p);
    mtfree(q);
    mtfree(dest);
}

void test_mttranspose_valid(void)
{
    Matrix *p = FILL(mtalloc(2, 3), 1, 2, 3, 4, 5, 6);
    Matrix *dest = mtalloc(3, 2);

    Matrix *result = mttranspose(p, dest);

    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL_DOUBLE(1.0, ENTRY(dest, 0, 0));
    TEST_ASSERT_EQUAL_DOUBLE(4.0, ENTRY(dest, 0, 1));
    TEST_ASSERT_EQUAL_DOUBLE(2.0, ENTRY(dest, 1, 0));
    TEST_ASSERT_EQUAL_DOUBLE(5.0, ENTRY(dest, 1, 1));
    TEST_ASSERT_EQUAL_DOUBLE(3.0, ENTRY(dest, 2, 0));
    TEST_ASSERT_EQUAL_DOUBLE(6.0, ENTRY(dest, 2, 1));

    mtfree(p);
    mtfree(dest);
}

void test_mttranspose_incompatible(void)
{
    Matrix *p = FILL(mtalloc(2, 3), 1, 2, 3, 4, 5, 6);
    Matrix *dest = mtalloc(2, 2); // Incompatible dimensions for transpose

    Matrix *result = mttranspose(p, dest);

    TEST_ASSERT_NULL(result);
    TEST_ASSERT_EQUAL_UINT32(MT_ERR_INCOMPATIBLE, mterrno);

    mtfree(p);
    mtfree(dest);
}

void test_mttranspose_square_matrix(void)
{
    Matrix *p = FILL(mtalloc(2, 2), 1, 2, 3, 4);
    Matrix *dest = mtalloc(2, 2);

    Matrix *result = mttranspose(p, dest);

    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL_DOUBLE(1.0, ENTRY(dest, 0, 0));
    TEST_ASSERT_EQUAL_DOUBLE(3.0, ENTRY(dest, 0, 1));
    TEST_ASSERT_EQUAL_DOUBLE(2.0, ENTRY(dest, 1, 0));
    TEST_ASSERT_EQUAL_DOUBLE(4.0, ENTRY(dest, 1, 1));

    mtfree(p);
    mtfree(dest);
}

void test_mttranspose_single_element(void)
{
    Matrix *p = FILL(mtalloc(1, 1), 1);
    Matrix *dest = mtalloc(1, 1);

    Matrix *result = mttranspose(p, dest);

    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL_DOUBLE(1.0, ENTRY(dest, 0, 0));

    mtfree(p);
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
    RUN_TEST(test_mtscale_valid);
    RUN_TEST(test_mtscale_incompatible);
    RUN_TEST(test_mtdivide_valid);
    RUN_TEST(test_mtdivide_incompatible);
    RUN_TEST(test_mtapply_valid);
    RUN_TEST(test_mtapply_incompatible);
    RUN_TEST(test_mtmult_null);
    RUN_TEST(test_mttranspose_valid);
    RUN_TEST(test_mttranspose_incompatible);
    RUN_TEST(test_mttranspose_square_matrix);
    RUN_TEST(test_mttranspose_single_element);
    return UNITY_END();
}
