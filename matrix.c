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
    }

    free(p->entries);
    free(p);
}

int mtsave(Matrix *p, const char *path)
{
    FILE *fp = fopen(path, "wb");

    if (fp == NULL)
        return -1;
    if (fwrite(&p->row, sizeof(int), 1, fp) != 1)
        goto error;
    if (fwrite(&p->col, sizeof(int), 1, fp) != 1)
        goto error;
    if (fwrite(p->entries, sizeof(double), p->row * p->col, fp) != p->row * p->col)
        goto error;
    
    fclose(fp);
    return 0;

    error:
    fclose(fp);
    return -2;
}

Matrix *mtload(const char *path)
{
    FILE *fp = fopen(path, "rb");
    int row, col;

    if (fp == NULL)
        return NULL;
    if (fread(&row, sizeof(int), 1, fp) != 1)
        goto error0;
    if (fread(&col, sizeof(int), 1, fp) != 1)
        goto error0;
    Matrix *p = mtalloc(row, col);

    if (fread(p->entries, sizeof(double), row * col, fp) != row * col)
        goto error1;
    fclose(fp);
    return p;

    error0:
    fclose(fp);
    return NULL;
    error1:
    fclose(fp);
    mtfree(p);
    return NULL;
}

Matrix *mtadd(Matrix *p, Matrix *q, Matrix *dest)
{
    if (p->row != q->row || p->col != q->col) {
        fprintf(stderr, "%s(%p, %p, %p): Incompatible matrices (%d, %d) and (%d, %d)\n", __func__, p, q, dest, p->row, p->col, q->row, q->col);
        mterrno |= MT_ERR_INCOMPATIBLE;
    }

    if (p->row != dest->row || p->col != dest->col) {
        fprintf(stderr, "%s(%p, %p, %p): Incompatible destination (%d, %d) for matrices (%d, %d) and (%d, %d)\n", __func__, p, q, dest, dest->row, dest->col, p->row, p->col, q->row, q->col);
        mterrno |= MT_ERR_INCOMPATIBLE;
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
    }

    if (p->row != dest->row || p->col != dest->col) {
        fprintf(stderr, "%s(%p, %p, %p): Incompatible destination (%d, %d) for matrices (%d, %d) and (%d, %d)\n", __func__, p, q, dest, dest->row, dest->col, p->row, p->col, q->row, q->col);
        mterrno |= MT_ERR_INCOMPATIBLE;
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
    }

    if (p->row != dest->row || p->col != dest->col) {
        fprintf(stderr, "%s(%p, %p, %p): Incompatible destination (%d, %d) for matrices (%d, %d) and (%d, %d)\n", __func__, p, q, dest, dest->row, dest->col, p->row, p->col, q->row, q->col);
        mterrno |= MT_ERR_INCOMPATIBLE;
    }

    for (int i = 0; i < p->row * p->col; i++)
        dest->entries[i] = p->entries[i] * q->entries[i];
    return dest;
}