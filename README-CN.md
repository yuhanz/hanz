# Hanz
使用汉字的神经网络 DSL (PyTorch 神经网络构建)

常见的深度学习项目通常以程序的方式构建神经网络。不同人有不同的编程习惯，造成同一个神经网络的构建可能看起来非常不同。需要读者看懂源码。程序表达的神经网络远不如图表直观。

Hanz 项目的目标是为神经网络提供 DSL，达到以下效果
- 编写好的神经网络与编程语言无关
- 易于理解且易于编写
- 编写好的神经网络本身展示一个图表。 （无需再使用工具构建图表）
- 可以轻松编辑、复制和粘贴神经网络的某一段、某一行
- 自动填充神经网络层之间所需的中间参数。

## 如何安装依赖的程序库
```
make setup
```

## 运行测试程序
```
make test
```

## 命令行显示一个hanz文件生成的神经网络
```
make file_name=examples/example-nerf-with-view-dir.hanz display
```

## 主要思路

用1个汉字作为神经网络层（算子）。 一个神经网络表示为一竖行汉字。每个汉字是一层，输出结果输入到下一层。有些汉字代表的算子可以把临近两列的结果组合成一列，例如：将。两个神经网络的结果相加，或首尾相连两个结果并成一个结果。

我选汉字代表的神经网络层（算子）的标注是：这个汉字的样子需要接近算子的含义，而和这个汉字的原义无关。 作为汉算子的字的读法也根据编程者的习惯，可以念汉字读音、日文汉字读音、广东话读音，也可以念出它的本意（比如直接念 ReLu）

### Why Chinese characters (Hanzi)?
Chinese characters are square characters, made to align well vertically. Traditionally Chinese is also written vertically. So the characters are very natural. Chinese characters meant to be pronunciation agnostic. It is intended to be pronounced as what local dialect wants to pronounce. Many Chinese characters are pictograms. The shape of the character means something. Here I changed its definition so that the shape close to its meaning for neural network domain. For example,
- 中 means InstanceNormalization, was normalization tries to center data.
- 田 looks like convolution kernel, and thus means convolution filter
- 井 is chosen to be convolution transpose, as the shape suggests kernel / image grid.
- 厂 is chosen to represent Relu operator.

### 为什么用汉字表示算子？

汉字方块字，竖行容易对齐。 古书中文也是竖写的，所以汉字天然适应作为的竖写的表达式。 同时一个汉字的意思与发音无关，可以根据方言来发音。作为象形文字，字的形状解释了它的含义，便于记忆。 在这里我故意改变了一个字的定义，让它的形状接近它在神经网络领域的含义。 例如，
- **中** 表示InstanceNormalization，是规范化尝试使数据居中。
- **田** 看起来像卷积核，因此表示卷积滤波器 Convolution
- **井** 看起来类似像素表，因为表示为 Convolution Transpose
- **厂** 如果把这个字转180度，很像 ReLU 的函数坐标图。

以下是一个简单的hello world 使用范例。这个这个例子定义了 Resnet 的片段:
输入是三个通道的2D图，执行卷积输出 256 个通道，然后归一化(Instance Normalization)，最后 ReLu
```
3
田 ... 256
中 ... 256
厂
```

以上会被解释为以下程序:
```
nn.Sequential(nn.Conv2d(3, 256), nn.InstanceNorm2d(256), nn.ReLU())
```

单个汉字能表达意思的有很多，所以一个字就可以表达一个算子，不需要用多字的词。

### 如果我需要用程序写神经网络怎么办？

可以用你最熟悉的模板引擎(template engine) 来生成一个 .hanz文件。Django, Jinja2, Ruby ERB 都可以. 然后你想循环还是使用变量、想生成什么样的神经网络就随你了。

下面是一个使用 Ruby ERB 建构 10层 Resnet 的例子:
```
3<% 10.times do %>
川田 ... 256
川中 ... 256
川厂
川田 ... 256
川中 ... 256
+<% end %>
```

### 我不会写怎么办？

可以复制粘贴。我举了很多使用的例子

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
