#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "network.h"

size_t total_weight_count(NetworkShape shape) {
  size_t prev_dim = shape.input_dim;
  size_t count = 0;
  for (size_t layer_i = 0; layer_i < shape.layer_count; layer_i++) {
    size_t curr_dim = shape.layer_dims[layer_i];
    count += prev_dim * curr_dim;
    prev_dim = curr_dim;
  }
  return count;
}

size_t total_neuron_count(NetworkShape shape) {
  size_t count = 0;
  for (size_t layer_i = 0; layer_i < shape.layer_count; layer_i++) {
    count += shape.layer_dims[layer_i];
  }
  return count;
}

NetworkVec create_network_vec(NetworkShape shape) {
  size_t weight_count = total_weight_count(shape);
  size_t neuron_count = total_neuron_count(shape);

  Scalar *weights_base = malloc(sizeof(Scalar) * weight_count);
  Scalar **neuron_weights_base = malloc(sizeof(Scalar*) * neuron_count);
  Scalar ***weights = malloc(sizeof(Scalar**) * shape.layer_count);
  size_t prev_dim = shape.input_dim;
  for (size_t layer_i = 0; layer_i < shape.layer_count; layer_i++) {
    size_t curr_dim = shape.layer_dims[layer_i];
    Scalar **layer_weights = neuron_weights_base;
    neuron_weights_base += curr_dim;
    for (size_t neuron_i = 0; neuron_i < curr_dim; neuron_i++) {
      layer_weights[neuron_i] = weights_base;
      weights_base += prev_dim;
    }
    weights[layer_i] = layer_weights;
    prev_dim = curr_dim;
  }

  Scalar *biases_base = malloc(sizeof(Scalar) * neuron_count);
  Scalar **biases = malloc(sizeof(Scalar*) * shape.layer_count);
  for (size_t layer_i = 0; layer_i < shape.layer_count; layer_i++) {
    biases[layer_i] = biases_base;
    biases_base += shape.layer_dims[layer_i];
  }

  NetworkVec net;
  net.weights = weights;
  net.biases = biases;
  return net;
}

// NOTE: This only works if `net` was allocated using create_network_vec
void free_network_vec(NetworkVec net) {
  free(net.weights[0][0]); // Actually frees ALL neuron weight arrays
  free(net.weights[0]); // Actually free ALL layer weight arrays
  free(net.weights);

  free(net.biases[0]); // Actually free ALL layer bias arrays
  free(net.biases);
}

SignalVec create_signal_vec(NetworkShape shape) {
  Scalar *signals_base = malloc(sizeof(Scalar) * total_neuron_count(shape));
  Scalar **signals = malloc(sizeof(Scalar*) * shape.layer_count);
  for (size_t layer_i = 0; layer_i < shape.layer_count; layer_i++) {
    signals[layer_i] = signals_base;
    signals_base += shape.layer_dims[layer_i];
  }

  SignalVec sig;
  sig.signals = signals;
  return sig;
}

// NOTE: This only works if `sig` was allocated using create_signal_vec
void free_signal_vec(SignalVec sig) {
  free(sig.signals[0]); // Actually frees ALL layer signal arrays
  free(sig.signals);
}

Scalar rand_scalar() {
  return (Scalar)rand() / (Scalar)RAND_MAX * 2.0 - 1;
}

void randomize_network(
  // Input parameters:
  NetworkShape shape,

  // Output parameters:
  NetworkVec net
) {
  size_t prev_dim = shape.input_dim;
  for (size_t layer_i = 0; layer_i < shape.layer_count; layer_i++) {
    size_t curr_dim = shape.layer_dims[layer_i];
    Scalar **layer_weights = net.weights[layer_i];
    Scalar *layer_biases = net.biases[layer_i];
    for (size_t neuron_i = 0; neuron_i < curr_dim; neuron_i++) {
      layer_biases[neuron_i] = rand_scalar();
      Scalar *neuron_weights = layer_weights[neuron_i];
      for (size_t weight_i = 0; weight_i < prev_dim; weight_i++) {
        neuron_weights[weight_i] = rand_scalar();
      }
    }
    prev_dim = curr_dim;
  }
}

void zero_network(
  // Input parameters:
  NetworkShape shape,

  // Output parameters:
  NetworkVec net
) {
  size_t prev_dim = shape.input_dim;
  for (size_t layer_i = 0; layer_i < shape.layer_count; layer_i++) {
    size_t curr_dim = shape.layer_dims[layer_i];
    Scalar **layer_weights = net.weights[layer_i];
    Scalar *layer_biases = net.biases[layer_i];
    for (size_t neuron_i = 0; neuron_i < curr_dim; neuron_i++) {
      layer_biases[neuron_i] = 0.0;
      Scalar *neuron_weights = layer_weights[neuron_i];
      for (size_t weight_i = 0; weight_i < prev_dim; weight_i++) {
        neuron_weights[weight_i] = 0.0;
      }
    }
    prev_dim = curr_dim;
  }
}

void feed_forward(
  // Input parameters:
  NetworkShape shape,
  NetworkVec net,
  Scalar *input,

  // Output parameters:
  SignalVec sig
) {
  size_t prev_dim = shape.input_dim;
  Scalar *prev_layer_sig = input;
  for (size_t layer_i = 0; layer_i < shape.layer_count; layer_i++) {
    size_t curr_dim = shape.layer_dims[layer_i];
    Scalar **layer_weights = net.weights[layer_i];
    Scalar *layer_biases = net.biases[layer_i];
    Scalar *curr_layer_sig = sig.signals[layer_i];
    for (size_t neuron_i = 0; neuron_i < curr_dim; neuron_i++) {
      Scalar *neuron_weights = layer_weights[neuron_i];
      Scalar neuron_signal = layer_biases[neuron_i];
      for (size_t weight_i = 0; weight_i < prev_dim; weight_i++) {
        neuron_signal += neuron_weights[weight_i] * prev_layer_sig[weight_i];
      }
      neuron_signal = tanh(neuron_signal);
      curr_layer_sig[neuron_i] = neuron_signal;
    }
    prev_dim = curr_dim;
    prev_layer_sig = curr_layer_sig;
  }
}

void back_propagate(
  // Input parameters:
  NetworkShape shape,
  NetworkVec net,
  SignalVec sig,
  Scalar *input,
  Scalar *target_output,
  Scalar eta, // Learning constant

  // Output parameters:
  SignalVec sig_grad, // Can be uninitialized

  // Will be *added to*, with the difference multiplied by eta.
  // It is safe for this to be the network itself.
  NetworkVec net_grad
) {
  // Compute signal gradient for output layer:
  {
    size_t output_layer_i = shape.layer_count - 1;
    size_t output_dim = shape.layer_dims[output_layer_i];
    Scalar *output_sig = sig.signals[output_layer_i];
    Scalar *output_grad = sig_grad.signals[output_layer_i];
    for (size_t neuron_i = 0; neuron_i < output_dim; neuron_i++) {
      output_grad[neuron_i] = 2.0 * (output_sig[neuron_i] - target_output[neuron_i]);
    }
  }

  for (size_t layer_i = shape.layer_count - 1; layer_i > 0; layer_i--) {
    // NOTE: Here, "previous" refers to the *input layer* to this layer

    // To read:
    size_t curr_dim = shape.layer_dims[layer_i];
    Scalar *curr_sig = sig.signals[layer_i];
    Scalar **curr_weights = net.weights[layer_i];
    Scalar *curr_sig_grad = sig_grad.signals[layer_i];

    size_t prev_dim = shape.layer_dims[layer_i - 1];
    Scalar *prev_sig = sig.signals[layer_i - 1];

    // to write:
    Scalar **curr_weight_grad = net_grad.weights[layer_i];
    Scalar *curr_bias_grad = net_grad.biases[layer_i];
    Scalar *prev_sig_grad = sig_grad.signals[layer_i - 1];

    // Zero out prev_sig_grad, since we're going to be cumulatively adding to it:
    for (size_t neuron_i = 0; neuron_i < prev_dim; neuron_i++) {
      prev_sig_grad[neuron_i] = 0.0;
    }

    for (size_t neuron_i = 0; neuron_i < curr_dim; neuron_i++) {
      Scalar neuron_sig = curr_sig[neuron_i];
      Scalar neuron_sig_grad = curr_sig_grad[neuron_i];
      Scalar *neuron_weights = curr_weights[neuron_i];
      Scalar *neuron_weight_grad = curr_weight_grad[neuron_i];

      // Derivative of the activation function of this neuron with respect to its argument
      Scalar theta_prime = 1 - neuron_sig * neuron_sig;

      // Compute bias gradient for this neuron:
      curr_bias_grad[neuron_i] += eta * neuron_sig_grad * theta_prime;

      for (size_t weight_i = 0; weight_i < prev_dim; weight_i++) {
        // Accumulate signal gradient for previous layer:
        prev_sig_grad[weight_i] += (neuron_sig_grad * theta_prime) * neuron_weights[weight_i];

        // Compute weight gradient for this neuron:
        neuron_weight_grad[weight_i] += (eta * neuron_sig_grad * theta_prime) * prev_sig[weight_i];
      }
    }
  }

  // Compute weight and bias gradients for first layer:
  {
    size_t first_dim = shape.layer_dims[0];
    Scalar *first_sig = sig.signals[0];
    Scalar *first_sig_grad = sig_grad.signals[0];
    Scalar **first_weights = net.weights[0];
    Scalar **first_weight_grad = net_grad.weights[0];
    Scalar *first_bias_grad = net_grad.biases[0];

    for (size_t neuron_i = 0; neuron_i < first_dim; neuron_i++) {
      Scalar neuron_sig = first_sig[neuron_i];
      Scalar neuron_sig_grad = first_sig_grad[neuron_i];
      Scalar *neuron_weight_grad = first_weight_grad[neuron_i];

      Scalar theta_prime = 1 - neuron_sig * neuron_sig;

      first_bias_grad[neuron_i] += eta * neuron_sig_grad * theta_prime;
      for (size_t weight_i = 0; weight_i < shape.input_dim; weight_i++) {
        neuron_weight_grad[weight_i] += (eta * neuron_sig_grad * theta_prime) * input[weight_i];
      }
    }
  }
}

void learn(
  // Input parameters:
  NetworkShape shape,
  Scalar *input,
  Scalar *target_output,
  Scalar eta,

  // Output parameters:
  NetworkVec net,
  SignalVec sig,
  SignalVec sig_grad
) {
  feed_forward(shape, net, input, sig);
  back_propagate(shape, net, sig, input, target_output, -eta, sig_grad, net);
}

void print_network(NetworkShape shape, NetworkVec net) {
  size_t prev_dim = shape.input_dim;
  printf("Network [layer count: %zu] [input dim: %zu]\n", shape.layer_count, shape.input_dim);
  for (size_t layer_i = 0; layer_i < shape.layer_count; layer_i++) {
    size_t curr_dim = shape.layer_dims[layer_i];
    Scalar **layer_weights = net.weights[layer_i];
    Scalar *layer_biases = net.biases[layer_i];
    printf("\n  Layer %zu [dim: %zu]\n", layer_i, curr_dim);
    for (size_t neuron_i = 0; neuron_i < curr_dim; neuron_i++) {
      Scalar bias = layer_biases[neuron_i];
      Scalar *neuron_weights = layer_weights[neuron_i];
      printf("    Neuron %zu [bias: %+0.6f]\n", neuron_i, bias);
      for (size_t weight_i = 0; weight_i < prev_dim; weight_i++) {
        printf("      Weight %zu: %+0.6f\n", weight_i, neuron_weights[weight_i]);
      }
    }
    prev_dim = curr_dim;
  }
}
