#include <stdio.h>
#include <stdint.h>

typedef struct Images {
  uint32_t count;
  uint32_t width;
  uint32_t height;
  uint32_t image_pixels;
  uint8_t *image_data;
} Images;

typedef struct Labels {
  uint32_t count;
  uint8_t *labels;
} Labels;

Images read_images(FILE *f);

Labels read_labels(FILE *f);

void free_images(Images images);

void free_labels(Labels labels);
