#pragma once
#include "nnetwork.h"

Dataset *read(const char *image_path, const char *label_path, int offset, int limit, int batch_size);
void print_image(Dataset *dataset);