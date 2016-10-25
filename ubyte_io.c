#include <stdlib.h>

#include "ubyte_io.h"

static uint32_t read_uint32(FILE *f) {
  uint8_t bytes[4];
  if (4 != fread(&bytes, sizeof(uint8_t), 4, f)) {
    puts("Could not read uint32 from file");
    exit(1);
  }
  return
    ((uint32_t)bytes[0] << 24) +
    ((uint32_t)bytes[1] << 16) +
    ((uint32_t)bytes[2] << 8) +
    (uint32_t)bytes[3];
}

Images read_images(FILE *f) {
  uint32_t magic_number = read_uint32(f);
  if (magic_number != 0x00000803) {
    printf("Incorrect magic number %x for image file\n", magic_number);
    exit(1);
  }

  uint32_t count = read_uint32(f);
  uint32_t width = read_uint32(f);
  uint32_t height = read_uint32(f);
  uint32_t image_pixels = width * height;

  uint32_t pixels = count * image_pixels;

  uint8_t *image_data = malloc(sizeof(uint8_t) * pixels);
  if (pixels != fread(image_data, sizeof(uint8_t), pixels, f)) {
    printf("Could not read images");
    exit(1);
  }

  Images image;
  image.count = count;
  image.width = width;
  image.height = height;
  image.image_pixels = image_pixels;
  image.image_data = image_data;
  return image;
}

Labels read_labels(FILE *f) {
  uint32_t magic_number = read_uint32(f);
  if (magic_number != 0x00000801) {
    printf("Incorrect magic number %x for label file\n", magic_number);
    exit(1);
  }

  uint32_t count = read_uint32(f);

  uint8_t *label_data = malloc(sizeof(uint8_t) * count);
  if (count != fread(label_data, sizeof(uint8_t), count, f)) {
    puts("Could not read labels");
    exit(1);
  }

  Labels labels;
  labels.count = count;
  labels.labels = label_data;

  return labels;
}

void free_images(Images images) {
  free(images.image_data);
}

void free_labels(Labels labels) {
  free(labels.labels);
}
