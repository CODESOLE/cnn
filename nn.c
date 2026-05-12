#include "nn.h"
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__aarch64__)
  // 64-bit Arm (Raspberry Pi 4/5 with 64-bit OS)
  #define DEBUG_BREAK() __asm__ __volatile__("brk #0")
#elif defined(__arm__)
  // 32-bit Arm (Older Pi or 32-bit OS)
  #define DEBUG_BREAK() __asm__ __volatile__("bkpt #0")
#elif defined(__i386__) || defined(__x86_64__)
  #define DEBUG_BREAK() __asm__ __volatile__("int $3")
#else
  #include <signal.h>
  #define DEBUG_BREAK() raise(SIGTRAP)
#endif

#define OUTPUT_VECTOR_SIZE 10
#define RNG 2

static inline float get_rand_uniform(const unsigned int n) {
  const float range = (float)(RAND_MAX % n);
  return (float)(rand() % n) - (range / 2.f);
}

struct NNData *nninit(float learn_rate, int layer_count, ...) {
  struct NNData *nn = malloc(sizeof(struct NNData));
  nn->curr_w = 0;
  nn->curr_b = 0;
  nn->learn_rate = learn_rate;
  struct Layer *l = malloc(sizeof(struct Layer) * layer_count);
  va_list args;
  va_start(args, layer_count);
  size_t prev_neuron_count;
  size_t total_weights = 0;
  size_t total_biases = 0;
  for (size_t i = 0; i < layer_count; ++i) {
    if (i == 0) {
      int neuron_counts = va_arg(args, int);
      l[i].neuron_sz = neuron_counts;
      l[i].n = malloc(sizeof(struct Neuron) * neuron_counts);
      for (size_t j = 0; j < l[i].neuron_sz; ++j) {
        l[i].n[j].a = get_rand_uniform(RNG);
        l[i].n[j].w = NULL;
        l[i].n[j].b = 0.f;
      }
      prev_neuron_count = neuron_counts;
    } else {
      int neuron_counts = va_arg(args, int);
      l[i].neuron_sz = neuron_counts;
      l[i].n = malloc(sizeof(struct Neuron) * neuron_counts);
      total_biases += neuron_counts;
      total_weights += prev_neuron_count * neuron_counts;
      for (size_t j = 0; j < l[i].neuron_sz; ++j) {
        l[i].n[j].a = get_rand_uniform(RNG);
        l[i].n[j].b = get_rand_uniform(RNG);
        l[i].n[j].w = malloc(prev_neuron_count * sizeof(float));
        for (size_t k = 0; k < prev_neuron_count; ++k) {
          l[i].n[j].w[k] = get_rand_uniform(RNG);
        }
      }
      prev_neuron_count = neuron_counts;
    }
  }
  va_end(args);
  nn->total_w = total_weights;
  nn->total_b = total_biases;
  nn->gradient = malloc((total_weights + total_biases) * sizeof(float));
  nn->layer_sz = layer_count;
  nn->layers = l;
  return nn;
}

#if 0
static inline float activation_fn(float x) { // ReLU
  return fmaxf(0.f, x);
}
static inline float deriv_activation_fn(float x) {
  return x > 0.f ? 1.f : 0.f;
}
#else
static inline float activation_fn(float x) { // SIGMOID
  return 1.f / (1.f + (1.f / expf(x)));
}
static inline float deriv_activation_fn(float x) {
  return activation_fn(x) * (1.f - activation_fn(x));
}
#endif

static float dC0_da(struct NNData *nn, float output_vector[OUTPUT_VECTOR_SIZE], size_t layer_idx, size_t neuron_idx) {
  assert(layer_idx < nn->layer_sz && "Given layer_idx exceeds NN total layer size");
  if (layer_idx == nn->layer_sz - 1) { // last layer 2(a_i - y_i)
    if (neuron_idx >= nn->layers[layer_idx].neuron_sz) DEBUG_BREAK();
    assert(neuron_idx < nn->layers[layer_idx].neuron_sz && "Given neuron_idx exceeds last layer neuron size");
    return 2.f * (nn->layers[layer_idx].n[neuron_idx].a - output_vector[neuron_idx]);
  } else { // hidden layers
    float dC_da = 0.f;
    for (size_t i = 0; i < nn->layers[layer_idx + 1].neuron_sz; ++i) {
      float z_j = 0.f;
      for (size_t j = 0; j < nn->layers[layer_idx].neuron_sz; ++j) { // z_j^(l+1)
        z_j += nn->layers[layer_idx + 1].n[i].w[j] * nn->layers[layer_idx].n[j].a;
      }
      dC_da += nn->layers[layer_idx + 1].n[i].w[neuron_idx] * deriv_activation_fn(z_j + nn->layers[layer_idx + 1].n[i].b) * dC0_da(nn, output_vector, layer_idx + 1, i);
    }
    return dC_da;
  }
  assert(0 && "UNREACHABLE");
}

static void dC0_db(struct NNData *nn, float output_vector[OUTPUT_VECTOR_SIZE], size_t layer_idx) {
  nn->curr_b = 0;
  for (size_t i = 0; i < nn->layers[layer_idx].neuron_sz; ++i) { // for every a1
    float z_j = 0.f;
    for (size_t j = 0; j < nn->layers[layer_idx - 1].neuron_sz; ++j) { // for every a0
      z_j += nn->layers[layer_idx].n[i].w[j] * nn->layers[layer_idx - 1].n[j].a;
    }
    float gb = deriv_activation_fn(z_j + nn->layers[layer_idx].n[i].b) * dC0_da(nn, output_vector, layer_idx, i);
    assert(nn->total_w + nn->curr_b < (nn->total_w + nn->total_b));
    nn->gradient[nn->total_w + nn->curr_b++] = gb;
  }
}

static void dC0_dw(struct NNData *nn, float output_vector[OUTPUT_VECTOR_SIZE], size_t layer_idx) {
  nn->curr_w = 0;
  float gw = 0.f;
  for (size_t i = 0; i < nn->layers[layer_idx].neuron_sz; ++i) { // for every a1
    float z_j = 0.f;
    for (size_t j = 0; j < nn->layers[layer_idx - 1].neuron_sz; ++j) { // for every a0
      z_j += nn->layers[layer_idx].n[i].w[j] * nn->layers[layer_idx - 1].n[j].a;
    }
    for (size_t j = 0; j < nn->layers[layer_idx - 1].neuron_sz; ++j) { // for every a0
      gw = nn->layers[layer_idx - 1].n[j].a * deriv_activation_fn(z_j + nn->layers[layer_idx].n[i].b) * dC0_da(nn, output_vector, layer_idx, i);
      assert(nn->curr_w < nn->total_w);
      nn->gradient[nn->curr_w++] = gw;
    }
  }
}

static float C0_cost(struct NNData *nn, float expected) {
  float cost = 0.f;
  float output_vector[OUTPUT_VECTOR_SIZE] = {0.f};
  output_vector[(size_t)expected] = 1.f;
  float err_diff = 0.f;
  for (size_t i = 0; i < nn->layers[nn->layer_sz - 1].neuron_sz; ++i) {
    err_diff = nn->layers[nn->layer_sz - 1].n[i].a - output_vector[i];
    err_diff *= err_diff;
    cost += (double)err_diff;
  }
  return cost;
}

static double calc_avg_cost(float expected, struct NNData *nn) {
  static float avg_cost = 0.f;
  avg_cost += C0_cost(nn, expected);
  return avg_cost;
}

double forward(struct NNData *nn, float *input, float expected) {
  for (size_t i = 0; i < nn->layers[0].neuron_sz; ++i) {
    nn->layers[0].n[i].a = input[i];
  }
  for (size_t i = 1; i < nn->layer_sz; ++i) { // a1 = activation_fn(W * a0 + b)
    for (size_t j = 0; j < nn->layers[i].neuron_sz; ++j) {
      float a = 0.f;
      for (size_t k = 0; k < nn->layers[i - 1].neuron_sz; ++k)
        a += nn->layers[i].n[j].w[k] * nn->layers[i - 1].n[k].a;
      nn->layers[i].n[j].a = activation_fn(a + nn->layers[i].n[j].b);
    }
  }
  return calc_avg_cost(expected, nn);
}

void backpropagation(struct NNData *nn, float expected) {
  float output_vector[OUTPUT_VECTOR_SIZE] = {0.f};
  output_vector[(size_t)expected] = 1.f;
  for (size_t i = nn->layer_sz - 1; i >= 1; i--)
    dC0_dw(nn, output_vector, i);
  for (size_t i = nn->layer_sz - 1; i >= 1; i--)
    dC0_db(nn, output_vector, i);
  nn->curr_w = 0;
  for (size_t i = nn->layer_sz - 1; i >= 1; --i) { // C(w) -= VC(w)
    for (size_t j = 0; j < nn->layers[i].neuron_sz; ++j) {
      for (size_t k = 0; k < nn->layers[i - 1].neuron_sz; ++k) {
        nn->layers[i].n[j].w[k] -= nn->learn_rate * nn->gradient[nn->curr_w++];
      }
    }
  }
  nn->curr_b = 0;
  for (size_t i = nn->layer_sz - 1; i >= 1; --i) { // C(b) -= VC(b)
    for (size_t j = 0; j < nn->layers[i].neuron_sz; ++j) {
        nn->layers[i].n[j].b -= nn->learn_rate * nn->gradient[nn->total_w + nn->curr_b++];
    }
  }
}

void save_model(struct NNData *nn, const char *fname) {
  assert(!strncmp(fname + strlen(fname) - 4, ".bin", strlen(".bin")) && "Model file should end with .bin extension!");
  size_t layer_size = nn->layer_sz;
  if (remove(fname) == 0) {
    printf("File removed successfully.\n");
  } else {
    printf("File could not removed or not exist!\n");
  }
  FILE *fp = fopen(fname, "wb");
  if (fp == NULL) {
    fprintf(stderr, "%s could not open!\n", fname);
    exit(1);
  }

  fwrite(&nn->layer_sz, sizeof(size_t), 1, fp);
  for (size_t i = 0; i < nn->layer_sz; ++i)
    fwrite(&nn->layers[i].neuron_sz, sizeof(size_t), 1, fp);

  for (size_t i = layer_size - 1; i >= 1; --i) {
    for (size_t j = 0; j < nn->layers[i].neuron_sz; ++j) {
      fwrite(nn->layers[i].n[j].w, nn->layers[i - 1].neuron_sz * sizeof(float), 1, fp);
    }
  }
  for (size_t i = layer_size - 1; i >= 1; --i) {
    for (size_t j = 0; j < nn->layers[i].neuron_sz; ++j) {
      fwrite(&nn->layers[i].n[j].b, sizeof(float), 1, fp);
    }
  }
  fclose(fp);
}

struct NNData *load_model(const char *fname, float learn_rate) {
  FILE *fp = fopen(fname, "rb");
  if (fp == NULL) {
    fprintf(stderr, "%s could not open!\n", fname);
    exit(1);
  }
  size_t layer_count;
  size_t neuron_counts;
  fread(&layer_count, sizeof(size_t), 1, fp);
  struct NNData *nn = malloc(sizeof(struct NNData));
  nn->curr_w = 0;
  nn->curr_b = 0;
  nn->learn_rate = learn_rate;
  struct Layer *l = malloc(sizeof(struct Layer) * layer_count);
  size_t prev_neuron_count;
  size_t total_weights = 0;
  size_t total_biases = 0;
  for (size_t i = 0; i < layer_count; ++i) {
    if (i == 0) {
      fread(&neuron_counts, sizeof(size_t), 1, fp);
      l[i].neuron_sz = neuron_counts;
      l[i].n = malloc(sizeof(struct Neuron) * neuron_counts);
      for (size_t j = 0; j < l[i].neuron_sz; ++j) {
        l[i].n[j].a = get_rand_uniform(RNG);
        l[i].n[j].w = NULL;
        l[i].n[j].b = 0.f;
      }
      prev_neuron_count = neuron_counts;
    } else {
      fread(&neuron_counts, sizeof(size_t), 1, fp);
      l[i].neuron_sz = neuron_counts;
      l[i].n = malloc(sizeof(struct Neuron) * neuron_counts);
      total_biases += neuron_counts;
      total_weights += prev_neuron_count * neuron_counts;
      for (size_t j = 0; j < l[i].neuron_sz; ++j) {
        l[i].n[j].a = get_rand_uniform(RNG);
        l[i].n[j].b = get_rand_uniform(RNG);
        l[i].n[j].w = malloc(prev_neuron_count * sizeof(float));
        for (size_t k = 0; k < prev_neuron_count; ++k) {
          l[i].n[j].w[k] = get_rand_uniform(RNG);
        }
      }
      prev_neuron_count = neuron_counts;
    }
  }
  nn->total_w = total_weights;
  nn->total_b = total_biases;
  nn->gradient = malloc((total_weights + total_biases) * sizeof(float));
  nn->layer_sz = layer_count;
  nn->layers = l;

  for (size_t i = layer_count - 1; i >= 1; --i) {
    for (size_t j = 0; j < nn->layers[i].neuron_sz; ++j) {
      fread(nn->layers[i].n[j].w, nn->layers[i - 1].neuron_sz * sizeof(float), 1, fp);
    }
  }
  for (size_t i = layer_count - 1; i >= 1; --i) {
    for (size_t j = 0; j < nn->layers[i].neuron_sz; ++j) {
      fread(&nn->layers[i].n[j].b, sizeof(float), 1, fp);
    }
  }

  fclose(fp);
  return nn;
}

void nndeinit(struct NNData *nn) {
  for (size_t i = 0; i < nn->layer_sz; ++i)
    for (size_t j = 0; j < nn->layers[i].neuron_sz; ++j)
      free(nn->layers[i].n[j].w);
  for (size_t i = 0; i < nn->layer_sz; ++i)
    free(nn->layers[i].n);
  free(nn->layers);
  free(nn->gradient);
  free(nn);
}
