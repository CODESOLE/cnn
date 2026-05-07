#ifndef __NN_H__
#define __NN_H__

#include <stddef.h>
#define CC_NO_SHORT_NAMES
#include "cc.h"

struct Neuron {
  float *w, b, a;
};

struct Layer {
  struct Neuron *n;
  size_t neuron_sz;
};

struct NNData {
  struct Layer *layers;
  size_t layer_sz;
  size_t total_w;
  size_t total_b;
  size_t curr_w; //< points to current weight to update in backpropagation
  size_t curr_b; //< points to current bias to update in backpropagation
  float *gradient;
  double learn_rate;
};

struct NNData *
nninit(float learn_rate, int count, ...);

double
forward(struct NNData *nn, float *input, float expected);

void
backpropagation(struct NNData *nn, float expected);

void
save_model(struct NNData *nn, const char *fname);

struct NNData *
load_model(const char *fname, float learn_rate);

void
nndeinit(struct NNData *nn);

#endif

