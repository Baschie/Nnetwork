# Nnetwork

This is a simple multi-layer neural network. The `mnistnet.c` file demonstrates how to use it by training a neural network on the MNIST dataset.

## Downloading the Dataset

You can download the MNIST dataset from this link: https://www.kaggle.com/datasets/hojjatk/mnist-dataset.

You might need to update these macros in `mnistnet.c` depending on where you decide to extract the files:
```C
#define TRAINIMAGES "archive/train-images.idx3-ubyte"
#define TRAINLABELS "archive/train-labels.idx1-ubyte"
#define TESTIMAGES  "archive/t10k-images.idx3-ubyte"
#define TESTLABELS  "archive/t10k-labels.idx1-ubyte"
```

## Compiling

Run the following command to build the executable named `mnistnet`:
```sh
make
```

## How to Use

Compile `matrix.c` and `nnetwork.c` and link them to your program.

### Creating a Neural Network

Use the `nnetalloc` function to create a neural network:
```C
Nnet *nnetalloc(int input_size, int *layer_sizes, Activation *functions, int nlay);
```

Example:
```C
Activation activation = {sigmoid, sigmoid_derivative};
Nnet *mnistnet = nnetalloc(300, (int[]) {200, 100, 10}, (Activation[]) {activation, activation, activation}, 3);
```

- `input_size`: Number of neurons in the input layer.
- `layer_sizes`: An array containing the number of neurons in each of the hidden and output layers.
- `functions`: Array of activation functions for each layer.
- `nlay`: Number of layers excluding the input layer, which is the length of `layer_sizes`.

Different layers can have different activation functions. You have to define the activation functions and their derivatives, ensuring they take a `double` argument and return a `double`, and put them inside an `Activation` struct object. The derivative receives the output of the activation function as its input during training.

### Creating a Dataset

To train the network, create a `Dataset` object. The `Dataset` struct is defined as below:
```C
typedef struct {
    int size;
    Matrix *inputs;
    Matrix *targets;
} Batch;

typedef struct {
    int nbatch;
    Batch *batches;
} Dataset;
```

- `nbatch`: Number of batches.
- `size`: Number of `inputs` and `targets` in the batch.

Matrix struct:
```C
typedef struct {
    int row;
    int col;
    double *entries;
} Matrix;
```

Targets are the expected predictions for inputs. Input and target matrices must have a column of 1. For memory efficiency in classification tasks, you can allocate and initialize a limited number of `double` arrays and have all target matrices' entries point to them.

### Training the Network

Train the network with the `stochastic_train` function:
```C
void stochastic_train(Nnet *nnet, Dataset *dataset, int epoches, double learning_rate);
```

In a dataset like MNIST with 60,000 images, each epoch takes about 2 minutes. Additional stuff in the program such as IO take about 3 minutes. With a learning rate of 0.05 the network reaches about 89% accuracy in the first epoch. After ten epochs, it'll reach about 96% accuracy.

#### The `set_epsilon` function
It's defined as:
```C
void set_epsilon(double x)
{
    epsilon = x;
}
```
`epsilon` is a variable used to prevent division by zero. Setting it to a too big or too small value might unstabilize the training. In the `mnistnet.c` file it's set to a value of 1e-5.
### Making Predictions

Use the network with the `predict` function:
```C
Matrix *predict(Nnet *nnet, Matrix *input, Matrix *dest);
```

- `dest`: This is the matrix where the result will be saved. The function returns a pointer to `dest`.

You can create a matrix using `mtalloc(int row, int col)` and free it with `mtfree(Matrix *p)`.

### Testing Accuracy

You can test the accuracy of your network after training using the `accuracy` function:
```C
double accuracy(Nnet *nnet, Dataset *dataset, double (*interpret)(Matrix *));
```

- `interpret`: Function that receives the output of `predict` and the target matrices in the dataset, and compares the values produced by each.

As you see it's more useful for classification tasks.
### Saving and Loading the Network

You can save the network to a file after training using `nnetsave`:
```C
void nnetsave(Nnet *nnet, const char *path);
```

And load it back with `nnetload`:
```C
Nnet *nnetload(const char *path, Activation *functions);
```

`nnetsave` does not store the activation functions of each layer, so you have to keep track of them yourself for when you load the network back from the file.