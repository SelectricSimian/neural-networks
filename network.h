typedef float Scalar;

typedef struct NetworkShape {
  size_t layer_count;
  size_t input_dim;
  size_t *layer_dims;
} NetworkShape;

typedef struct NetworkVec {
  Scalar ***weights;
  Scalar **biases;
} NetworkVec;

typedef struct SignalVec {
  Scalar **signals;
} SignalVec;

size_t total_weight_count(NetworkShape shape);

size_t total_neuron_count(NetworkShape shape);

NetworkVec create_network_vec(NetworkShape shape);

// NOTE: This only works if `net` was allocated using create_network_vec
void free_network_vec(NetworkVec net);

SignalVec create_signal_vec(NetworkShape shape);

// NOTE: This only works if `sig` was allocated using create_signal_vec
void free_signal_vec(SignalVec sig);

void randomize_network(
  // Input parameters:
  NetworkShape shape,

  // Output parameters:
  NetworkVec net
);

void zero_network(
  // Input parameters:
  NetworkShape shape,

  // Output parameters:
  NetworkVec net
);

void feed_forward(
  // Input parameters:
  NetworkShape shape,
  NetworkVec net,
  Scalar *input,

  // Output parameters:
  SignalVec sig
);

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
);

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
);

void print_network(NetworkShape shape, NetworkVec net);
