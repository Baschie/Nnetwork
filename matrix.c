#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>

uint32_t mterrno;

Matrix *mtalloc(int row, int col)
{
    Matrix *p = malloc(sizeof(Matrix));

    if (p == NULL || (p->entries = malloc(sizeof(double) * row * col)) == NULL) {
        fprintf(stderr, "mtalloc(%d, %d): malloc returned NULL\n", row, col);
        mterrno |= MT_ERR_ALLOC;
        free(p);
        return NULL;
    }

    p->row = row;
    p->col = col;
    return p;
}

void mtfree(Matrix *p)
{
    if (p == NULL) {
        fprintf(stderr, "mtfree: freeing NULL Matrix attempted\n");
        mterrno |= MT_ERR_NULL_MATRIX;
        return;
    }

    free(p->entries);
    free(p);
}

int mtsave(Matrix *p, FILE *fp)
{
    if (p == NULL) {
        fprintf(stderr, "%s(%p, %p): Null matrix\n", __func__, p, fp);
        mterrno |= MT_ERR_NULL_MATRIX;
        return -1;
    }
    if (fp == NULL) {
        fprintf(stderr, "%s(%p, %p): Null file pointer\n", __func__, p, fp);
        mterrno |= MT_ERR_NULL_FP;
        return -1;
    }
    if (fwrite(&p->row, sizeof(int), 1, fp) != 1 ||
        fwrite(&p->col, sizeof(int), 1, fp) != 1 ||
        fwrite(p->entries, sizeof(double), p->row * p->col, fp) != p->row * p->col) {
        fprintf(stderr, "%s(%p, %p): Failed to save matrix\n", __func__, p, fp);
        mterrno |= MT_ERR_FILE_IO;
        return -1;
    }
    
    return 0;
}

Matrix *mtload(Matrix *dest, FILE *fp) /* Do not allocate memory for dest's entries */
{
    if (fp == NULL) {
        fprintf(stderr, "%s(%p, %p): Null file pointer\n", __func__, dest, fp);
        mterrno |= MT_ERR_NULL_FP;
        return NULL;
    }
    if (dest == NULL) {
        fprintf(stderr, "%s(%p, %p): Null matrix\n", __func__, dest, fp);
        mterrno |= MT_ERR_NULL_MATRIX;
        return NULL;
    }
    if (fread(&dest->row, sizeof(int), 1, fp) != 1 ||
        fread(&dest->col, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "%s(%p, %p): Failed to read row/col\n", __func__, dest, fp);
        mterrno |= MT_ERR_FILE_IO;
        return NULL;
    }
    dest->entries = malloc(sizeof(double) * dest->col * dest->row);
    if (fread(dest->entries, sizeof(double), dest->row * dest->col, fp) != dest->row * dest->col) {
        fprintf(stderr, "%s(%p, %p): Failed to read all entries of matrix\n", __func__, dest, fp);
        mterrno |= MT_ERR_FILE_IO;
        return NULL;
    }

    return dest;
}

Matrix *mtadd(Matrix *p, Matrix *q, Matrix *dest)
{
    if (p->row != q->row || p->col != q->col) {
        fprintf(stderr, "%s(%p, %p, %p): Incompatible matrices (%d, %d) and (%d, %d)\n", __func__, p, q, dest, p->row, p->col, q->row, q->col);
        mterrno |= MT_ERR_INCOMPATIBLE;
        return NULL;
    }

    if (p->row != dest->row || p->col != dest->col) {
        fprintf(stderr, "%s(%p, %p, %p): Incompatible destination (%d, %d) for matrices (%d, %d) and (%d, %d)\n", __func__, p, q, dest, dest->row, dest->col, p->row, p->col, q->row, q->col);
        mterrno |= MT_ERR_INCOMPATIBLE;
        return NULL;
    }

    for (int i = 0; i < p->row * p->col; i++)
        dest->entries[i] = p->entries[i] + q->entries[i];
    return dest;
}

Matrix *mtsubtract(Matrix *p, Matrix *q, Matrix *dest)
{
    if (p->row != q->row || p->col != q->col) {
        fprintf(stderr, "%s(%p, %p, %p): Incompatible matrices (%d, %d) and (%d, %d)\n", __func__, p, q, dest, p->row, p->col, q->row, q->col);
        mterrno |= MT_ERR_INCOMPATIBLE;
        return NULL;
    }

    if (p->row != dest->row || p->col != dest->col) {
        fprintf(stderr, "%s(%p, %p, %p): Incompatible destination (%d, %d) for matrices (%d, %d) and (%d, %d)\n", __func__, p, q, dest, dest->row, dest->col, p->row, p->col, q->row, q->col);
        mterrno |= MT_ERR_INCOMPATIBLE;
        return NULL;
    }

    for (int i = 0; i < p->row * p->col; i++)
        dest->entries[i] = p->entries[i] - q->entries[i];
    return dest;
}

Matrix *mtelmult(Matrix *p, Matrix *q, Matrix *dest)
{
    if (p->row != q->row || p->col != q->col) {
        fprintf(stderr, "%s(%p, %p, %p): Incompatible matrices (%d, %d) and (%d, %d)\n", __func__, p, q, dest, p->row, p->col, q->row, q->col);
        mterrno |= MT_ERR_INCOMPATIBLE;
        return NULL;
    }

    if (p->row != dest->row || p->col != dest->col) {
        fprintf(stderr, "%s(%p, %p, %p): Incompatible destination (%d, %d) for matrices (%d, %d) and (%d, %d)\n", __func__, p, q, dest, dest->row, dest->col, p->row, p->col, q->row, q->col);
        mterrno |= MT_ERR_INCOMPATIBLE;
        return NULL;
    }

    for (int i = 0; i < p->row * p->col; i++)
        dest->entries[i] = p->entries[i] * q->entries[i];
    return dest;
}

Matrix *mtscale(Matrix *p, double scaler, Matrix *dest)
{
    if (p->row != dest->row || p->col != dest->col) {
        fprintf(stderr, "%s(%p, %lf, %p): Incompatible matrices (%d, %d) and (%d, %d)\n", __func__, p, scaler, dest, p->row, p->col, dest->row, dest->col);
        mterrno |= MT_ERR_INCOMPATIBLE;
        return NULL;
    }

    for (int i = 0; i < p->row * p->col; i++)
        dest->entries[i] = p->entries[i] * scaler;
    return dest;
}

Matrix *mtdivide(Matrix *p, double divisor, Matrix *dest)
{
    if (p->row != dest->row || p->col != dest->col) {
        fprintf(stderr, "%s(%p, %lf, %p): Incompatible matrices (%d, %d) and (%d, %d)\n", __func__, p, divisor, dest, p->row, p->col, dest->row, dest->col);
        mterrno |= MT_ERR_INCOMPATIBLE;
        return NULL;
    }

    for (int i = 0; i < p->row * p->col; i++)
        dest->entries[i] = p->entries[i] / divisor;
    return dest; 
}

Matrix *mtapply(Matrix *p, double (*func)(double), Matrix *dest)
{
    if (p->row != dest->row || p->col != dest->col) {
        fprintf(stderr, "%s(%p, %p, %p): Incompatible matrices (%d, %d) and (%d, %d)\n", __func__, p, func, dest, p->row, p->col, dest->row, dest->col);
        mterrno |= MT_ERR_INCOMPATIBLE;
        return NULL;
    }

    for (int i = 0; i < p->row * p->col; i++)
        dest->entries[i] = func(p->entries[i]);
    return dest;
}

Matrix *mtmult(Matrix *p, Matrix *q, Matrix *dest)
{
    if (p == NULL || q == NULL || dest == NULL) {
        fprintf(stderr, "%s(%p, %p, %p): One or more matrices are NULL\n", __func__, p, q, dest);
        mterrno |= MT_ERR_NULL_MATRIX;
        return NULL;
    }

    if (p->col != q->row) {
        fprintf(stderr, "%s(%p, %p, %p): Incompatible matrices (%d, %d) and (%d, %d)\n", __func__, p, q, dest, p->row, p->col, q->row, q->col);
        mterrno |= MT_ERR_INCOMPATIBLE;
        return NULL;
    }

    if (p->row != dest->row || q->col != dest->col) {
        fprintf(stderr, "%s(%p, %p, %p): Incompatible destination (%d, %d) for matrices (%d, %d) and (%d, %d)\n", __func__, p, q, dest, dest->row, dest->col, p->row, p->col, q->row, q->col);
        mterrno |= MT_ERR_INCOMPATIBLE;
        return NULL;
    }

    for (int i = 0; i < p->row; i++) {
        for (int j = 0; j < q->col; j++) {
            ENTRY(dest, i, j) = 0;
            for (int k = 0; k < p->col; k++) {
                ENTRY(dest, i, j) += ENTRY(p, i, k) * ENTRY(q, k, j);
            }
        }
    }
    return dest;
}

Matrix *mttranspose(Matrix *p, Matrix *dest)
{
    if (p->row != dest->col || p->col != dest->row) {
        fprintf(stderr, "%s(%p, %p): Incompatible matrices (%d, %d) and (%d, %d)\n", __func__, p, dest, p->row, p->col, dest->row, dest->col);
        mterrno |= MT_ERR_INCOMPATIBLE;
        return NULL;
    }

    for (int i = 0; i < p->row; i++)
        for (int j = 0; j < p->col; j++)
            ENTRY(dest, j, i) = ENTRY(p, i, j);
    return dest;
}