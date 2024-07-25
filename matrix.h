#pragma once
#include <string.h>

#define ENTRY(mat, i, j) mat->entries[mat->col * i + j]
#define FILL(mat, ...) ({Matrix *_mat = mat; memcpy(_mat->entries, (double []) {__VA_ARGS__}, \
	sizeof((double []) {__VA_ARGS__})); _mat;}) 

typedef struct {
	int row;
	int col;
	double *entries;
} Matrix;

Matrix *mtalloc(int row, int col);
void mtfree(Matrix *p);
int mtsave(Matrix *p, const char *path);