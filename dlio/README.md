# DLIO benchmark

The benchmark is designed to measure the performance of training and evaluation of deep learning models on data stored
as HDF5 files. Based on collected and analysed I/O patterns from [DLIO Benchmark](https://github.com/argonne-lcf/dlio_benchmark),
this benchmark simulates the learning process and evaluation of deep learning models that use PyTorch and Tensorflow
frameworks, while gathering valuable information about system performance. Most importantly, this extension allows users
to test AI workloads without the need to install machine learning libraries, reducing complexity and enhancing the
usability of the benchmark. Another advantage is that from our experiments, our extension ran faster than DLIO Benchmark,
which we suspect was due to the difference in the overhead introduced by the C application in our extension and the
Python application in the original benchmark. While the quicker runtime could be beneficial for faster testing, it also
suggests that the benchmark might not fully capture the complexity of real AI workloads, such as high metadata
operations introduced by the use of Python-based libraries. I/O pattern produced by this extension is based on the
implementation of [DLIO benchmark version 1.1](https://github.com/argonne-lcf/dlio_benchmark/releases/tag/v1.1).
Changes in the main DLIO Benchmark configurations after version 1.1 will not be reflected in this h5bench pattern. To
reproduce them, DLIO Benchmark behavior can be studied using various I/O analysis tools. We recommend using 
[Log VFD](https://docs.hdfgroup.org/hdf5/v1_14/group___f_a_p_l.html#ga4e03be2fe83ed02b32266a6c81427beb).


## Configuration
As in the case with other extensions, the following parameters should be specified in the configuration section of the 
json file to configure the benchmark:

| Parameter              | Description                                                         | Type   | Default  |
|------------------------|---------------------------------------------------------------------|--------|----------|
| generate-data          | Enable generation of benchmarking data                              | bool   | false    |
| train                  | Enable model training simulation                                    | bool   | false    |
| evaluation             | Enable model evaluation simulation                                  | bool   | false    |
| record-length          | Record size of a single sample in bytes                             | int    | 67108864 |
| num-files-train        | The number of files used to train the model                         | int    | 32       |
| num-files-eval         | The number of files used to evaluate the model                      | int    | 8        |
| num-samples-per-file   | The number of samples in each file                                  | int    | 4        |
| data-folder            | Name of the directory storing the benchmark data                    | string | ./data   |
| file-prefix            | Prefix in the name of files containing training and evaluation data | string | img      |
| chunking               | Enable chunking                                                     | bool   | false    |
| chunk-size             | Chunk size                                                          | int    | 1024     |
| keep-files             | Does not delete data after the benchmark is finished                | bool   | false    |
| compression            | Enable compression                                                  | bool   | false    |
| compression-level      | Compression level from 1 to 9                                       | int    | 4        |
| batch-size             | Training batch size                                                 | int    | 7        |
| batch-size-eval        | Evaluation batch size                                               | int    | 2        |
| shuffle                | Enable samples shuffle                                              | bool   | false    |
| preprocess-time        | Preprocessing time after reading each sample in seconds             | float  | 0.0      |
| preprocess-time-stdev  | Standard deviation in preprocessing time in seconds                 | float  | 0.0      |
| epochs                 | The number of epochs                                                | int    | 5        |
| total-training-steps   | Maximum number of steps per training per epoch                      | int    | -1       |
| computation-time       | Computation time after reading each batch in seconds                | float  | 0.323    |
| computation-time-stdev | Standard deviation in computation time in seconds                   | float  | 0.0      |
| random-seed            | Random seed to be used                                              | int    | 42       |
| eval-time              | Evaluation time after reading each batch in seconds                 | float  | 0.323    |
| eval-time-stdev        | Standard deviation in evaluation time in seconds                    | float  | 0.0      |
| epochs-between-evals   | The number of epochs between evaluations                            | int    | 1        |
| train-data-folder      | Name of the directory containing the training data                  | string | train    |
| valid-data-folder      | Name of the directory containing the validation data                | string | valid    |
| records-dataset-name   | Name of the dataset with records                                    | string | records  |
| labels-dataset-name    | Name of the dataset with labels                                     | string | labels   |
| seed-change-epoch      | Enable seed changes every epoch                                     | bool   | false    |
| read-threads           | The number of workers used to read the data                         | int    | 4        |
| collective-meta        | Enable collective HDF5 metadata operations                          | bool   | false    |
| collective-data        | Enable collective HDF5 data operations                              | bool   | false    |
| subfiling              | Enable HDF5 Subfiling Virtual File Driver                           | bool   | false    |
| output-csv-name        | Name of the output csv file                                         | string | output   |
| output-ranks-data      | Enable statistics output for each rank                              | bool   | false    |

It should be noted that for each parameter there is a default value that applies if the parameter has not been specified 
in the configuration file. Thus, by default the benchmark will not run because the generate-data, train and evaluation 
parameters are false. A sample configuration file can be found in the `samples/` directory.

## Understanding the output
The sample output of the benchmark is as follows:
```
=================== Performance Results ==================
Total number of ranks: 8
The number of read threads per rank: 0
Total training set size: 7.000 GB
Training set size per rank: 896.000 MB
Total training emulated compute time: 3.229 s
Training metadata time: 2.808 s
Training raw read time: 30.905 s
Training average raw read rate: 145.141 MB/s
Observed training completion time: 37.432 s
Observed average training rate: 131.044 MB/s
Training average throughput: 1.871 samples/s
Training throughput standard deviation: 0.037 samples/s
Training average IO: 119.729 MB/s
Training IO standard deviation: 2.379 MB/s
Total evaluation set size: 7.000 GB
Evaluation set size per rank: 896.000 MB
Total evaluation emulated compute time: 3.206 s
Evaluation metadata time: 2.805 s
Evaluation raw read time: 31.699 s
Evaluation average raw read rate: 141.906 MB/s
Observed evaluation completion time: 38.424 s
Observed average evaluation rate: 127.595 MB/s
Evaluation average throughput avg: 1.826 samples/s
Evaluation throughput standard deviation: 0.090 samples/s
Evaluation average IO: 116.883 MB/s
Evaluation IO standard deviation: 5.735 MB/s
===========================================================
```
Let's take a closer look at it. First, information about the number of MPI ranks and processes per MPI rank used in the 
simulation is output. Then, the same values are used to describe the training and evaluation performance, so for the 
sake of reducing redundancy, let us consider only the first half of the results concerning the training process. Total 
training set size is calculated as the size of all HDF5 files used for training. Accordingly, the training set size per 
rank gives an idea of how much of the load is taken over by one MPI rank. Total training emulated compute time contains 
information about the total time spent on compute emulation for all epochs in total, as well as training metadata time 
and training raw read time, about which, however, it should be noted that they are not interleaved and measure the time 
of execution of `H5Fopen`, `H5Dget_space`, `H5Screate_simple`, `H5Sclose` and `H5Dread` commands respectively. Training 
average raw read rate is calculated as training set size per rank divided by training raw read time. Observed training 
completion time includes all the time spent on the training process, among other things including resource allocation 
and computation simulation. Observed average training rate is equal to training set size per rank divided by the 
difference of observed training completion time and total training emulated compute time, thus showing the data reading 
rate without taking into account emulation costs. Training average throughput and training throughput standard deviation 
give an indication of the number of samples from the training dataset processed in one second. Training average IO and 
Training IO standard deviation translate these values into bytes/second by multiplying by the size of one sample.

## Future work

There are plans to add more configuration options for the extension in the future to increase its flexibility:
- Add settings for Subfiling VFD. Currently, the default settings are used.
- Add more features from [DLIO Benchmark](https://github.com/argonne-lcf/dlio_benchmark) such as resizable records.
- Analyze and add support for other ml frameworks and data loaders. For example, DALI.
- Add support for prefetching.
- Expand the ability to randomly shuffle samples. At the moment, it is not possible to shuffle only samples in each file
without changing the order of the files for training.
- Add more compression filters and thus support different compression algorithms for HDF5 data.
- Add support for drop_last customization. Currently, by default, all batches left after MPI ranks distribution are not processed.
- Replace the use of `fork()` with `MPI_Comm_spawn()` when creating new processes, as using `fork()` with MPI may be unsafe
- Test support for the Cache VOL connector.
- Add support for checkpointing by saving the model to a hdf5 file.
