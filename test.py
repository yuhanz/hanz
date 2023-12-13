import hanz
import torch.nn as nn
import torch
import numpy as np
import functools

def pipelineToString(pipeline):
    return ",".join(map(lambda m: m._get_name(), pipeline))

def test_numberOfParameters():
    reused = hanz.parseHanzLines("""10
    森 ... 1
    川川
    +
    """.split("\n"))
    assert sum([p.shape.numel() for p in reused[0].parameters()]) == 11

    two = hanz.parseHanzLines("""10
    森川 ... 1
    川森 ... 1
    +
    """.split("\n"))
    assert sum([p.shape.numel() for p in two[0].parameters()]) == 22



def test_example():
    modules = hanz.parseHanz('examples/example.hanz')
    assert len(modules) == 1
    assert isinstance(modules[0], nn.Module)
    assert pipelineToString(modules[0]) == "Linear,LeakyReLU,ReLU,Sigmoid,Softplus,Softmax,Sin"
    input = torch.Tensor([[1,2]])
    output = modules[0](input)
    assert len(output) == 1
    assert len(output[0]) == 4

def test_multiple_module_pipeliness():
    modules = hanz.parseHanz('examples/example2a.hanz')
    assert len(modules) == 2
    assert isinstance(modules[0], nn.Module)
    assert isinstance(modules[1], nn.Module)
    assert pipelineToString(modules[0]) == "Sin,Linear,ReLU"
    assert pipelineToString(modules[1]) == "Linear,LeakyReLU,ReLU,Sigmoid,Softplus,Softmax,Sin"

def test_combine_multiple_module_pipes():
    modules = hanz.parseHanz('examples/example2.hanz')
    assert len(modules) == 1
    assert isinstance(modules[0], hanz.CustomCombine)
    assert pipelineToString(modules[0].fn1) == "Sin,Linear,ReLU"
    assert pipelineToString(modules[0].fn2) == "Linear,LeakyReLU,ReLU,Sigmoid,Softplus,Softmax,Sin"

def test_pineline_with_concatenation():
    modules = hanz.parseHanz('examples/example-nerf-part1.hanz')
    assert len(modules) == 1
    m = modules[0]
    assert isinstance(m, hanz.CustomCombine)
    x = torch.rand(3, 12)
    result = m(x)
    num_tests, dim  = result.shape
    assert num_tests == 3
    assert dim == 562

def test_generator():
    modules = hanz.parseHanz('examples/example-generator.hanz')
    assert len(modules) == 1
    m = modules[0]
    x = torch.ones([2,3,9,9])
    result = m(x)
    num_tests, channels, width, height  = result.shape
    print("num_tests", num_tests)
    print("channels", channels)
    print("width", width)
    print("height", height)
    assert num_tests == 2
    assert channels == 3
    assert width == 6
    assert height == 6

def test_discriminator():
    modules = hanz.parseHanz('examples/example-discriminator.hanz')
    assert len(modules) == 1
    m = modules[0]
    x = torch.ones([2,3,9,9])
    result = m(x)
    num_tests, dim  = result.shape
    print("num_tests", num_tests)
    print("dim", dim)
    assert num_tests == 2
    assert dim == 1


def test_pineline_with_one_working_operator_in_row():
    modules = hanz.parseHanz('examples/example-nerf-with-view-dir.hanz')
    assert len(modules) == 1
    m = modules[0]
    x = torch.rand(1, 28)
    result = m(x)
    num_tests, dim  = result.shape
    assert num_tests == 1
    assert dim == 4

def test_multiple_input():
    modules, functions = hanz.parseHanz('examples/example-multiple-input.hanz')
    assert len(functions) == 2
    assert len(modules) == 2
    f = functions[0]
    f2 = functions[1]
    assert type(modules[0]) is torch.nn.modules.container.Sequential
    assert type(modules[1]) is torch.nn.modules.container.Sequential
    assert type(f) is functools.partial
    assert type(f2) is functools.partial
    assert pipelineToString(modules[0]) == 'SelectColumns,Sin,LeakyReLU,ReLU'
    assert pipelineToString(modules[1]) == 'SelectColumns,Cos'
    result = f(embedding = torch.ones(1,5) * 0.5, position = torch.ones(1,2) * 0.5)
    result2 = f2(embedding = torch.ones(1,5) * 0.5, position = torch.ones(1,2) * 0.5)
    assert result.tolist() == [[0.4794255495071411, 0.4794255495071411, 0.4794255495071411, 0.4794255495071411, 0.4794255495071411]], 'expecting {} received {}'.format(expected_result, result)
    assert result2.tolist() == [[0.8775825500488281, 0.8775825500488281]], 'expecting {} received {}'.format(expected_result, result2)

def test_repeat_columns():
    modules, functions = hanz.parseHanz('examples/example-decoder.hanz')
    assert len(functions) == 1
    assert len(modules) == 1
    f = functions[0]
    assert type(modules[0]) is hanz.hanz.CustomCombine
    assert type(f) is functools.partial
    assert pipelineToString(nn.Sequential(modules[0])) == 'Add'  # TODO: make this test case better
    result = f(query_vector = torch.ones(1,2) * 0.5, word_embedding = torch.ones(1,2) * 0.5, positional_encoding = torch.ones(1,2) * 0.5)
    assert result.tolist() == [[0.4794255495071411, 0.4794255495071411, 0.4794255495071411, 0.4794255495071411, 0.4794255495071411]], 'expecting {} received {}'.format(expected_result, result)
