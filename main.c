#define CC_NO_SHORT_NAMES
#include "cc.h"
#include "csv.h"
#include "nn.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

cc_vec(float) parse_file(const char *path) {
  assert(path);
  const int max_line_size = 4096;
  int done, err = 0;
  FILE *fp = fopen(path, "r");
  if (!fp) {
    fprintf(stderr, "Error occured while openning %s file!\n", path);
    exit(1);
  }
  cc_vec(float) dataset;
  cc_init(&dataset);
  char *line = NULL;
  char *tok = NULL;
  uint8_t label = 0;
  do {
    line = fread_csv_line(fp, max_line_size, &done, &err);
    if (!line || line[0] == '\0')
      break;
    if (err) {
      fprintf(stderr, "Error occured while parsing mnist_train.csv: %d!\n", err);
      exit(1);
    }
    tok = strtok(line, ",");
    assert(tok != NULL);
    label = atoi(tok);
    cc_push(&dataset, (float)label);
    tok = strtok(NULL, ",");
    while (tok != NULL) {
      cc_push(&dataset, (float)atoi(tok) / 255.f); // 28*28=784+1=785
      tok = strtok(NULL, ",");
    }
    free(line);
  } while (!done);
  fclose(fp);
  return dataset;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr,
            "Program should be run with either: cnn.out <train_data.csv> "
            "<model_paramaters_save_ouput.bin> or cnn.out <test.csv> "
            "<model_input_paramater.bin>");
    exit(1);
  }
  if (strncmp(argv[1] + strlen(argv[1]) - 4, ".bin", strlen(".bin")) == 0) {
	double avgcost = 0.0;
    struct NNData *nn = load_model(argv[1], 1.f);
    cc_vec(float) test_dataset = parse_file(argv[2]);
    size_t test_dataset_sz = cc_size(&test_dataset);
    size_t test_dataset_count = test_dataset_sz / 785;

    int64_t idx = -1;
    float biggest_activation = 0.f;
    size_t correct_count = 0;
    for (size_t i = 0; i < test_dataset_sz; i += 785) {
      size_t current = (i / 785) + 1;
      float label = *cc_get(&test_dataset, i);
      avgcost = forward(nn, cc_get(&test_dataset, i + 1), label);
      for (size_t j = 0; j < nn->layers[nn->layer_sz - 1].neuron_sz; j++) {
	if (nn->layers[nn->layer_sz - 1].n[j].a > biggest_activation) {
	  idx = j;
	  biggest_activation = nn->layers[nn->layer_sz - 1].n[j].a;
	}
      }
      assert(idx > -1 && "Index should not be negative");
      if (idx == (size_t)label)
	correct_count++;
      printf("[%zu / %zu] %%%f\n", correct_count, current, 100.f * ((float)correct_count / (float)current));
    }

    cc_cleanup(&test_dataset);
    nndeinit(nn);
  } else {
    srand(time(NULL));
    struct NNData *nn = nninit(1e-3, 4, 784, 16, 16, 10);
    cc_vec(float) train_dataset = parse_file(argv[1]);
    size_t train_dataset_sz = cc_size(&train_dataset);
    size_t train_dataset_count = train_dataset_sz / 785;
    double avgcost = 0.0;

    for (size_t i = 0; i < train_dataset_sz ; i += 785) {
      size_t current = (i / 785) + 1;
      printf("Training in-progress...[%zu / %zu] %%%f\n", current, train_dataset_count, 100.f * ((float)current / (float)train_dataset_count));
      float label = *cc_get(&train_dataset, i);
      avgcost = forward(nn, cc_get(&train_dataset, i + 1), label);
      backpropagation(nn, label);
    }
    printf("\n\n==========TRANING DONE==========\n\n");
    save_model(nn, argv[2]);
    printf("\n\n==========MODEL PARAMATERS WRITTEN TO DISK==========\n\n");
    /* printf("Avg Cost of NN: %g\n", avgcost / (double)train_dataset_count); */

    cc_cleanup(&train_dataset);
    nndeinit(nn);
  }

  return 0;
}
