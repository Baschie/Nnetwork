#pragma once
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#define ENTRY(mat, i, j) mat->entries[mat->col * i + j]
#define FILL(mat, ...) ({Matrix *_mat = mat; memcpy(_mat->entries, (double []) {__VA_ARGS__}, \
	sizeof((double []) {__VA_ARGS__})); _mat;}) 

/* Error codes */
#define MT_ERR_ALLOC (1 << 0)
#define MT_ERR_NULL_MATRIX (1 << 1)
#define MT_ERR_INCOMPATIBLE (1 << 2)
#define MT_ERR_NULL_FP (1 << 3)
#define MT_ERR_FILE_IO (1 << 4)

typedef struct {
	int row;
	int col;
	double *entries;
} Matrix;

extern uint32_t mterrno;

Matrix *mtalloc(int row, int col);
void mtfree(Matrix *p);
int mtsave(Matrix *p, FILE *fp);
Matrix *mtload(Matrix *dest, FILE *fp); /* Do not allocate memory for dest's entries */
Matrix *mtadd(Matrix *p, Matrix *q, Matrix *dest);
Matrix *mtsubtract(Matrix *p, Matrix *q, Matrix *dest);
Matrix *mtelmult(Matrix *p, Matrix *q, Matrix *dest); /* Element wise matrix multiplication */
Matrix *mtscale(Matrix *p, double scaler, Matrix *dest);
Matrix *mtdivide(Matrix *p, double divisor, Matrix *dest);
Matrix *mtapply(Matrix *p, double (*func)(double), Matrix *dest);
Matrix *mtmult(Matrix *p, Matrix *q, Matrix *dest);
Matrix *mttranspose(Matrix *p, Matrix *dest);