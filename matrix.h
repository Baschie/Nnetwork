#pragma once
#include <string.h>

typedef struct {
	int row;
	int col;
	double *entries;
} Matrix;

#define ENTRY(mat, i, j) mat->entries[mat->col * i + j]
#define FILL(mat, ...) mat->memcpy(mat->entries, (double []) {__VA_ARGS__}, \
	sizeof((double []) {__VA_ARGS__})/sizeof(double))