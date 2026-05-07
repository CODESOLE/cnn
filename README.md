# cnn
simple MLP based DNN in C for learning and exploration purposes.

# Compile
```console
make
```

# Usage
For training first parameter must be traning dataset second paramater must be trained model output file name with .bin extension
```console
cnn <train_data.csv> <model_paramaters_save_ouput.bin>
```

For testing first parameter must be model binary that ends with .bin extension and second paramater must be test data
```console
cnn.out <test.csv> <model_input_paramater.bin>
```

# LICENCE
MIT
