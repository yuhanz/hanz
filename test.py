import hanz
import torch.nn as nn
import torch

def pipelineToString(pipeline):
    return ",".join(map(lambda m: m._get_name(), pipeline))

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

def test_pineline_with_one_working_operator_in_row():
    modules = hanz.parseHanz('examples/example-nerf-with-view-dir.hanz')
    assert len(modules) == 1
    m = modules[0]
    x = torch.rand(1, 18)
    result = m(x)
    num_tests, dim  = result.shape
    assert num_tests == 1
    assert dim == 3
