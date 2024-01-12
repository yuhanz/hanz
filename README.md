# Hanz
Neural Network DSL using Hanzi (Chinese Characters) to build Neural Network in PyTorch

Deep learning projects usually construct neural networks programmatically.
The construction of the same neural network could look very different when written by two different people. It will require the viewer to understand the code.
It is hard to communicate the idea without a diagram.

The goal of the Hanz project is to provide a DSL for neural network so that
- It is language agnostic
- It is easy to understood and easy to write
- It represents a diagram itself. (No need to use a tool to construct a diagram)
- It can be easily edited, copied and pasted
- Automatically fill in the required intermediate parameters between the layers of neural network.

## How to Install

```
pip install hanz
```

## How to Use

```
import hanz
import torch
modules = hanz.parseHanz('examples/example-generator.hanz')
model = modules[0]
x = torch.ones([2,3,9,9])
result = model(x)
print(result)
```


## To install dependencies from this project
```
make setup
```

## To Run test cases
```
make test
```

## To display a hanz file
```
make file_name=examples/example-nerf-with-view-dir.hanz display
```

## Idea

Use 1 Chinese character as a neural network layer (operator). A neural network is expressed as pipelining a series of layers vertically. Each column represents a pipeline.  Columns can be combined with operators, for example: adding the results of two pipelines or feature concatenation.

A neural network layer (operator) is assigned to a Chinese character whose shape is close to what the operator does. The original meaning of the Chinese character doesn't matter. And it is up to the user to decide its pronunciation.

### Why Chinese characters (Hanzi)?
Chinese characters are square characters, made to align well vertically. Traditionally Chinese is also written vertically. So the characters fit the vertical flow very naturally. Chinese characters meant to be pronunciation agnostic. It is intended to be pronounced as what local dialect wants to pronounce. Many Chinese characters are pictograms. The shape of the character means something. Here I changed its definition so that the shape close to its meaning for neural network domain. For example,
- 中 means InstanceNormalization, was normalization tries to center data.
- 田 looks like convolution kernel, and thus means convolution filter
- 井 is chosen to be convolution transpose, as the shape suggests kernel / image grid.
- 厂 is chosen to represent Relu operator.

Here is hello world example of a fragment of a Resnet in Hanz. This performs a convolution to output 256 channels, apply instant normalization, and taking a ReLu.
```
3
田 ... 256
中 ... 256
厂
```

This translates to
```
nn.Sequential(nn.Conv2d(3, 256), nn.InstanceNorm2d(256), nn.ReLU())
```

There are many Chinese characters to choose from, so that it doesn't need multiple characters to mean an operator.

### But what if I want to construct programmatically?

Use your favorite template engine to construct a hanz file - Django, Jinja2, Ruby ERB. You may do your for-loop, variable replacement in a template to make a neural network arbitrary size.

Here is an example of Resnet repeating 10 times with Ruby ERB:
```
3<% 10.times do %>
川田 ... ; 256
川中 ... ; 256
川厂
川田 ... ; 256
川中 ... ; 256
+<% end %>
```

### But I cannot type Chinese.

Copy the operators from examples.

## Syntax, format and convention.

A Hanz file is parsed into 1 neural network as a function. The result is basically a pytorch neural network. So you can apply what you do in pytorch like calling a function . For example, transforming an input of 10 records with of 9 features as a Tensor of shape (10, 9).

The first line: of the file indicates the dimension(s) of 1 record.

The neural network definition starts from the second line. It assumes the input is 1 pipeline of records. When there are more vertical pipelines in the next line than the previous line. It assumes the newly added column connects to the identical instance of right most pipeline.

In the example below, the result from LeakyReLU (广) pipeline in the next row is becoming two pipelines: one identity operator (川), and one applied convolution (田).
```
广 ... 0.2
川田 ... ; 256
```

Note that a semicolon `;` separates parameters for different operator because there are 2 pipelines per row. When multiple parameters exist for the same pipeline, separate them with comma `,`.

It is possible to further configure a torch.nn operator with optional named parameters.
Here is an example of configuring a `torch.nn.Conv2d` module with detailed optional parameters for convolution kernel_size, padding, bias, etc:

```
田 ... 64, kernel_size = (3,3),  padding = 1, bias=False
```


### Operators and Meaning

| Operator    | Description  |Number of Input Pipelines | Parameters | Named Parameters (optional) |
| ----------- | ----------- | ------------------------- | ---------- | ---------------- |
|森 | Linear (fully connected) layer| 1 | (**output_dimensions** :Int)  | named parameters of torch.nn.Linear
|厂 | ReLU       | 1 | | named parameters of torch.nn.ReLU
|广 | LeakyReLU        | 1 | (**threshold** :Float) | named parameters of torch.nn.LeakyReLU
|中 | Instance Normalization        | 1 | | named parameters of torch.nn.InstanceNorm2d
|申 | Batch Normalization        | 1 | | named parameters of torch.nn.BatchNorm2d
|川 | Identity Operator        | 1 | | |
|又 | Reference Pipeline on the Left, add 1 pipeline (with the same instance) | 0 |
|了 | Signoid Activation Function      | 1 | |
|丁 | Tanh Activation Function        | 1 | |
|亢 | Softmax Activation Function        | 1 | (**dim** :Int)
|扎 | Softplus Activation Function        | 1 | | named parameters of torch.nn.Softmax
|弓 | Sine Function        | 1 | |
|引 | Cosine Function        | 1 | |
|凶 | Absolute Value        | 1 | |
|风 | Square Root        | 1 | |
|一 | Flatten        | 1 | (**output_dimensions** :Int, **dim0** :Int optional, **dim1** :Int optional) |
|正 | Matrix Transpose	| 1 | (**dim0** :Int, **dim1** :Int)
|田 | Convolution        | 1 | (**output_dim**: Int) | named parameters of torch.nn.Conv2d
|井 | Convolution Transpose        | 1 | (**output_dim**: Int) | named parameters of torch.nn.ConvTranspose2d
|吕 | Select Columns        | 1 | (**start_index** :Int, **end_index** :Int) |
|昌 | Matrix Multiplication with a Matrix as learnable parameters| 1 | (**output_dimensions** :Int) |
|朋 | Matrix Multiplication of two neighboring pipelines| 2 | |
|羽 | Multiplication of two neighboring pipelines | 2 | |
|非 | Dot Product two neighboring pipelines | 2 | |
|+  | Sum of two adjacent columns, and merge into 1 column        | 2 | |
|十 | Sum of two adjacent columns, and merge into 1 column        | 2 | |
|艹 | Concatenation of two adjacent columns, and merge into 1 column        | 2 | |
|土 | Sum vertically. Add all rows together to make 1 row | 1 | |
|日 | Take 1 row and ignore all other rows | 1 |  (**row_index** :Int) |


### Multiple Inputs with Named Parameters

It is possible to pass multiple inputs in named parameters as input a Hanz model, and instead of assuming the input as 1 vector, separate the columns by the parameter names.

Here is an example of passing two variables as input: *query_vector* of 2 dimensions, *positional_encoding* of 2 dimensions, and word_embedding of 2 dimensions.

Specify which variables you'd like to each pipeline to start with in the second line. It is possible to let the multiple pipelines to start with the same variables. (If the second did not specify the columns, then the input would become 1 vector as the concatenation of all input variables. In that case, you can use 吕 operator to select columns yourself.)

```
query_vector:2 positional_encoding:2 word_embedding:2
|query_vector|positional_encoding|word_embedding|query_vector|positional_encoding|
日日川川川 ... -1 ; -1
...
```

When named parameters are used in the input, the output will include both the operators and the functions that wraps around the operators with the named parameter as input. Use the function to pass multiple parameters to the operator. And the operator is the corresponding the Pytorch model under the function.

For example, construct a model with multiple inputs, along with associated function to call. Take the parameters from the models to train and invoke the models by passing tensors through the named parameters.
```
models, functions = hanz.parseHanzLines("""pos:2 embedding:4
  |embedding|pos|pos|embedding
  """.split("\n"))

model = models[0]
func = functions[0]
optimizer = torch.optim.Adam(list(model.parameters())
func(pos= pos_tensor, embedding= embedding_tensor)
```

This example will return 4 models and 4 corresponding functions. To call the model, invoke the function with `func(pos= pos_tensor, embedding= embedding_tensor)`. The model is just the Pytorch model.
