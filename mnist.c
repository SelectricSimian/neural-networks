#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "network.h"
#include "ubyte_io.h"

// Hyperparameters:
#define LAYERS (3)
#define LABELS (10)
#define ITERATIONS (30000000)
#define ETA (0.005)
size_t layer_dims[LAYERS] = {20, 10, LABELS};

size_t index_of_max(size_t count, Scalar *values) {
  Scalar max_value = values[0];
  size_t max_index = 0;
  for (size_t i = 1; i < count; i++) {
    if (values[i] > max_value) {
      max_value = values[i];
      max_index = i;
    }
  }
  return max_index;
}

void learn_mnist(Images images, Labels labels, NetworkShape *shape_out, NetworkVec *net_out) {
  if (images.count != labels.count) {
    printf("Got %d images but %d labels", images.count, labels.count);
    exit(1);
  }

  NetworkShape shape;
  shape.layer_count = LAYERS;
  shape.layer_dims = layer_dims;
  shape.input_dim = images.image_pixels;

  NetworkVec net = create_network_vec(shape);
  randomize_network(shape, net);
  SignalVec sig = create_signal_vec(shape);
  SignalVec sig_grad = create_signal_vec(shape);

  Scalar *input = malloc(sizeof(Scalar) * images.image_pixels);
  Scalar *target_output = malloc(sizeof(Scalar) * LABELS);

  for (size_t label_i = 0; label_i < LABELS; label_i++) {
    target_output[label_i] = -1.0;
  }

  printf("Training for %d iterations\n", ITERATIONS);

  for (uint32_t i = 0; i < ITERATIONS; i++) {
    uint32_t example_i = rand() % images.count;

    uint8_t *image_base = images.image_data + example_i * images.image_pixels;
    for (uint32_t pixel_i = 0; pixel_i < images.image_pixels; pixel_i++) {
      input[pixel_i] = (Scalar)image_base[pixel_i] / 256.0;
    }

    uint8_t label = labels.labels[example_i];
    target_output[label] = +1.0;

    learn(shape, input, target_output, ETA, net, sig, sig_grad);

    target_output[label] = -1.0;
  }

  free_signal_vec(sig);
  free_signal_vec(sig_grad);
  free(input);
  free(target_output);


  if (shape_out != NULL) {
    *shape_out = shape;
  }

  if (net_out != NULL) {
    *net_out = net;
  }
}

uint32_t test_mnist(NetworkShape shape, NetworkVec net, Images images, Labels labels) {
  if (images.count != labels.count) {
    printf("Got %d images but %d labels", images.count, labels.count);
    exit(1);
  }

  if (images.image_pixels != shape.input_dim) {
    printf(
      "Images provided have %d pixels, but network has %zu inputs\n",
      images.image_pixels,
      shape.input_dim
    );
    exit(1);
  }

  SignalVec sig = create_signal_vec(shape);
  SignalVec sig_grad = create_signal_vec(shape);

  Scalar *input = malloc(sizeof(Scalar) * images.image_pixels);
  Scalar *target_output = malloc(sizeof(Scalar) * LABELS);

  uint32_t misclassifications = 0;
  for (uint32_t example_i = 0; example_i < images.count; example_i++) {
    uint8_t *image_base = images.image_data + example_i * images.image_pixels;
    for (uint32_t pixel_i = 0; pixel_i < images.image_pixels; pixel_i++) {
      input[pixel_i] = (Scalar)image_base[pixel_i] / 255.0;
    }

    uint8_t label = labels.labels[example_i];

    feed_forward(shape, net, input, sig);

    uint8_t predicted_label = index_of_max(LABELS, sig.signals[LAYERS - 1]);
    if (predicted_label != label) {
      misclassifications++;
    }
  }

  free_signal_vec(sig);
  free_signal_vec(sig_grad);
  free(input);
  free(target_output);

  return misclassifications;
}

Images load_images(const char *fname) {
  FILE *image_file = fopen(fname, "r");
  if (image_file == NULL) {
    printf("Could not open image file %s\n", fname);
    exit(1);
  }
  Images images = read_images(image_file);
  fclose(image_file);

  return images;
}

Labels load_labels(const char *fname) {
  FILE *label_file = fopen(fname, "r");
  if (label_file == NULL) {
    printf("Could not open file %s\n", fname);
    exit(1);
  }
  Labels labels = read_labels(label_file);
  fclose(label_file);

  return labels;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Expected 2 command line arguments, got %d\n", argc - 1);
    exit(1);
  }

  Images images = load_images(argv[1]);
  printf("Loaded %d images with dimensions %dx%d\n", images.count, images.width, images.height);

  Labels labels = load_labels(argv[2]);
  printf("Loaded %d labels\n", labels.count);

  NetworkShape shape;
  NetworkVec net;
  learn_mnist(images, labels, &shape, &net);

  puts("Trained network successfully");
  print_network(shape, net);

  puts("Testing network on training set...");
  uint32_t training_misclassifications = test_mnist(shape, net, images, labels);
  printf(
    "Misclassifications for training set: %d / %d -- %0.2f%% error rate\n",
    training_misclassifications,
    images.count,
    (float)training_misclassifications / (float)images.count * 100.0
  );

  free_images(images);
  free_labels(labels);

  if (argc >= 5) {
    puts("Loading test set...");

    Images test_images = load_images(argv[3]);
    printf("Loaded %d test images\n", test_images.count);

    Labels test_labels = load_labels(argv[4]);
    printf("Loaded %d test labels\n", test_labels.count);

    puts("Testing network on test set...");
    uint32_t test_misclassifications = test_mnist(shape, net, test_images, test_labels);
    printf(
      "Misclassifications for test set: %d / %d -- %0.2f%% error rate",
      test_misclassifications,
      test_images.count,
      (float)test_misclassifications / (float)test_images.count * 100.0
    );

    free_images(test_images);
    free_labels(test_labels);
  }

  free_network_vec(net);
  return 0;
}
