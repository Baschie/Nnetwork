#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

Matrix *mtalloc(int row, int col)
{
    Matrix *p = malloc(sizeof(Matrix));

    if (p == NULL) {
        fprintf(stderr, "mtalloc(%d, %d): malloc returned NULL\n", row, col);
        exit(EXIT_FAILURE);
    }

    p->row = row;
    p->col = col;
    p->entries = malloc(sizeof(double) * row * col);
    return p;
}

void mtfree(Matrix *p)
{
    if (p == NULL) {
        fprintf(stderr, "mtfree: freeing NULL Matrix attempted\n");
        exit(EXIT_FAILURE);
    }

    free(p->entries);
    free(p);
}