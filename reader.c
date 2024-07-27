#include <stdio.h>
#include <stdlib.h>
#include "reader.h"
#include "matrix.h"

#define LABELBYTE 0x00000801
#define IMAGEBYTE 0x00000803

double *target_entries[] = {(double []) {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    (double []) {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
    (double []) {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
    (double []) {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
    (double []) {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    (double []) {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
    (double []) {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
    (double []) {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
    (double []) {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
    (double []) {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};

Dataset *read(const char *image_path, const char *label_path, int offset, int limit, int batch_size)
{
    int c;
    FILE *imagefp, *labelfp;
    union {
        int number;
        unsigned char bytes[4];
    } integer;

    if ((imagefp = fopen(image_path, "rb")) == NULL) {
        fprintf(stderr, "Failed to open file %s\n", image_path);
        exit(1);
    }

    if ((labelfp = fopen(label_path, "rb")) == NULL) {
        fprintf(stderr, "Failed to open file %s\n", label_path);
        exit(1);
    }

    for (int i = 0; i < 4; i++) { /* reading the image header */
        if ((c = fgetc(imagefp)) == EOF) {
            fprintf(stderr, "Corrupted file %s\n", image_path);
            exit(2);
        }

        integer.bytes[3 - i] = c;
    }

    if (integer.number != IMAGEBYTE) {
        fprintf(stderr, "%s is not an image file\n", image_path);
        exit(2);
    }

    for (int i = 0; i < 4; i++) { /* reading the number of images */
        if ((c = fgetc(imagefp)) == EOF) {
            fprintf(stderr, "Corrupted file %s\n", image_path);
            exit(2);
        }

        integer.bytes[3 - i] = c;
    }

    int imageno = integer.number;

    for (int i = 0; i < 4; i++) { /* reading the number of rows */
        if ((c = fgetc(imagefp)) == EOF) {
            fprintf(stderr, "Corrupted file %s\n", image_path);
            exit(2);
        }

        integer.bytes[3 - i] = c;
    }

    int rows = integer.number;

    for (int i = 0; i < 4; i++) { /* reading the number of cols */
        if ((c = fgetc(imagefp)) == EOF) {
            fprintf(stderr, "Corrupted file %s\n", image_path);
            exit(2);
        }

        integer.bytes[3 - i] = c;
    }

    int cols = integer.number;

    for (int i = 0; i < 4; i++) { /* reading the label header */
        if ((c = fgetc(labelfp)) == EOF) {
            fprintf(stderr, "Corrupted file %s\n", label_path);
            exit(2);
        }

        integer.bytes[3 - i] = c;
    }

    if (integer.number != LABELBYTE) {
        fprintf(stderr, "%s is not a label file\n", label_path);
        exit(2);
    }

    for (int i = 0; i < 4; i++) { /* reading the number of labels */
        if ((c = fgetc(labelfp)) == EOF) {
            fprintf(stderr, "Corrupted file %s\n", label_path);
            exit(2);
        }

        integer.bytes[3 - i] = c;
    }

    int labelno = integer.number;

    if (imageno != labelno) {
        fprintf(stderr, "The number of images and labels doesn't match\n");
        exit(3);
    }

    if (offset + limit > imageno) {
        fprintf(stderr, "The specified offset and limit exceed the number of items in the files\n");
        exit(3);
    }

    if (limit % batch_size > 0) {
        fprintf(stderr, "Batchsize incompatible with the number of images\n");
        exit(1);
    }

    Dataset *dataset = malloc(sizeof(Dataset));
    dataset->nbatch = limit / batch_size;
    dataset->batches = malloc(sizeof(Batch) * dataset->nbatch);
    Matrix *images = malloc(sizeof(Matrix) * limit);
    Matrix *targets = malloc(sizeof(Matrix) * limit);
    double *pixels = malloc(sizeof(double) * rows * cols * limit);

    if (fseek(labelfp, offset, SEEK_CUR)) { /* skip the labels before offset */
        fprintf(stderr, "Skipping to offset failed in file %s\n", label_path);
        exit(4);
    }

    if (fseek(imagefp, offset * rows * cols, SEEK_CUR)) { /* skip the images before offset */
        fprintf(stderr, "Skipping to offset failed in file %s\n", image_path);
        exit(4);
    }

    unsigned char pixel_bytes[rows * cols];

    for (int i = 0; i < limit; i++) {
        if (i % batch_size == 0) {
            dataset->batches[i/batch_size].inputs = &images[i];
            dataset->batches[i/batch_size].targets = &targets[i];
            dataset->batches[i/batch_size].size = batch_size;
        }

        images[i].row = rows * cols;
        images[i].col = 1;
        targets[i].row = 10;
        targets[i].col = 1;

        if ((c = fgetc(labelfp)) == EOF) {
            fprintf(stderr, "Unexpectedly reached the end of file %s\n", label_path);
            exit(2);
        }

        targets[i].entries = target_entries[c];
        images[i].entries = pixels + i * rows * cols;
        if (fread(pixel_bytes, sizeof(unsigned char), rows * cols, imagefp) != rows * cols) {
            fprintf(stderr, "Failed to read all pixels from file %s\n", image_path);
            exit(2);
        }

        for (int j = 0; j < rows * cols; j++) {
            images[i].entries[j] = pixel_bytes[j] / 255.0;
        }
    }

    fclose(imagefp);
    fclose(labelfp);

    return dataset;
}

char toAscii(double scale)
{
    char shades[] = "@#%$+*;:,. ";
    return shades[(int) (scale * (sizeof(shades) - 2))];
}

void print_image(Dataset *dataset)
{
    for (int imagei = 0; imagei < dataset->batches->size; imagei++) {
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                putchar(toAscii(dataset->batches->inputs[imagei].entries[i * 28 + j]));
            }
            putchar('\n');
        }

        double **target = target_entries;
        while (*target != dataset->batches->targets[imagei].entries)
            target++;
        printf("Label: %d\n", target - target_entries);
    }
}