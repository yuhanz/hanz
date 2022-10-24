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

## How to use

```
import hanz
modules = hanz.parseHanz('examples/example-generator.hanz')
model = modules[0]
x = torch.ones([2,3,9,9])
result = model(x)
print(result)
```

## To install dependencies
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
Chinese characters are square characters, made to align well vertically. Traditionally Chinese is also written vertically. So the characters are very natural. Chinese characters meant to be pronunciation agnostic. It is intended to be pronounced as what local dialect wants to pronounce. Many Chinese characters are pictograms. The shape of the character means something. Here I changed its definition so that the shape close to its meaning for neural network domain. For example,
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
川田 ... 256
川中 ... 256
川厂
川田 ... 256
川中 ... 256
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
川田 ... 256
```

### Operators and Meaning

| Operator    | Description |
| ----------- | ----------- |
|森 | Linear (fully connected) layer|
|厂 | ReLU       |
|广 | LeakyReLU        |
|中 | Instance Normalization        |
|申 | Batch Normalization        |
|川 | Identity Operator        |
|了 | Signoid Activation Function      |
|丁 | Tanh Activation Function        |
|亢 | Softmax Activation Function        |
|扎 | Softplus Activation Function        |
|弓 | Sine Function        |
|引 | Cosine Function        |
|凶 | Absolute Value        |
|风 | Square Root        |
|一 | Flatten        |
|吕 | Select Columns        |
|田 | Convolution        |
|井 | Convolution Transpose        |
|+  | Sum of two adjacent columns, and merge into 1 column        |
|十 | Sum of two adjacent columns, and merge into 1 column        |
|艹 | Concatenation of two adjacent columns, and merge into 1 column        |
