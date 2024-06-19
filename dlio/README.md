# DLIO benchmark

The benchmark is designed to measure the performance of training and evaluation of deep learning models on data stored as HDF5 files.

## Overview

### Command-line Arguments (Options)

- ``--generate-data``: Enable generation of benchmarking data. [default: *false*]
- ``--train``: Enable model training simulation [default: *false*]
- ``--evaluation``: Enable model evaluation simulation [default: *false*]
- ``--record-length <x>``: Record size of a single sample in bytes [default: *67108864*]
- ``--num-files-train <x>``: The number of files used to train the model [default: *64*]
- ``--num-files-eval <x>``: The number of files used to evaluate the model [default: *8*]
- ``--num-samples-per-file <x>``: The number of samples in each file [default: *4*]
- ``--data-folder <x>``: Name of the directory storing the benchmark data [default: *./data*]
- ``--file-prefix <x>``: Prefix in the name of files containing training and evaluation data [default: *img*]
- ``--chunking``: Enable chunking [default: *false*]
- ``--chunk-size <x>``: Chunk size [default: *1024*]
- ``--keep-files``: Does not delete data after the benchmark is finished [default: *1024*]
- ``--compression``: Enable compression [default: *false*]
- ``--compression-level <x>``: Compression level from 1 to 9 [default: *4*]
- ``--batch-size <x>``: Training batch size [default: *7*]
- ``--batch-size-eval <x>``: Evaluation batch size [default: *2*]
- ``--shuffle``: Enable samples shuffle [default: *false*]
- ``--preprocess-time <x>``: Preprocessing time after reading each sample in seconds [default: *0.0*]
- ``--preprocess-time-stdev <x>``: Standard deviation in preprocessing time in seconds [default: *0.0*]
- ``--epochs <x>``: The number of epochs [default: *5*]
- ``--computation-time <x>``: Computation time after reading each batch in seconds [default: *0.323*]
- ``--computation-time-stdev <x>``:  Standard deviation in computation time in seconds [default: *0.0*]
- ``--random-seed <x>``: Random seed to be used [default: *42*]
- ``--eval-time <x>``: Evaluation time after reading each batch in seconds [default: *0.323*]
- ``--eval-time-stdev <x>``: Standard deviation in evaluation time in seconds [default: *0.0*]
- ``--epochs-between-evals <x>``: The number of epochs between evaluations [default: *1*]
- ``--train-data-folder <x>``: Name of the directory containing the training data [default: *train*]
- ``--valid-data-folder <x>``: Name of the directory containing the validation data [default: *valid*]
- ``--records-dataset-name <x>``: Name of the dataset with records [default: *records*]
- ``--labels-dataset-name <x>``: Name of the dataset with labels [default: *labels*]
- ``--seed-change-epoch``: Enable seed changes every epoch [default: *false*]
- ``--read-threads``: The number of workers used to read the data [default: *4*]

### Exerciser Basics

## Building Exerciser
