import hanz
import torch.nn as nn
import torch
import numpy as np
import functools

def pipelineToString(pipeline):
    return ",".join(map(lambda m: m._get_name(), pipeline))

def test_numberOfParameters():
    # 1 network reused twice. So there is still 11 parameters
    reused = hanz.parseHanzLines("""10
    森 ... 1
    川川
    +
    """.split("\n"))
    assert sum([p.shape.numel() for p in reused[0].parameters()]) == 11

    # 2 networks are defined. So there are 11 * 2 parameters
    two = hanz.parseHanzLines("""10
    森川 ... 1
    川森 ... 1
    +
    """.split("\n"))
    assert sum([p.shape.numel() for p in two[0].parameters()]) == 22

def test_operator_to_duplicate_pipelines_with_identity_operator():
    operators = hanz.parseHanzLines("""3
    川森 ... ; 2
    森川 ... 1
    """.split("\n"))
    assert operators[0].in_features == 3
    assert operators[0].out_features == 1
    assert operators[1].in_features == 3
    assert operators[1].out_features == 2

    operators = hanz.parseHanzLines("""3
    森 ... 2
    森川川 ... 1
    """.split("\n"))
    assert len(operators) == 3
    assert isinstance(operators[0], nn.Sequential)
    assert isinstance(operators[1], nn.Linear)
    assert len(operators[0]) == 2
    assert operators[0][0].in_features == 3
    assert operators[0][0].out_features == 2
    assert operators[0][1].in_features == 2
    assert operators[0][1].out_features == 1
    assert operators[1].in_features == 3
    assert operators[1].out_features == 2
    assert operators[2].in_features == 3
    assert operators[2].out_features == 2


def test_operators_without_params():
    operators = hanz.parseHanzLines("""2
    厂广了丁扎弓引凶风土 ... ; 0.5 ;
    """.split("\n"))
    assert len(operators) == 10
    assert operators[0](torch.Tensor([[-1,2]])).tolist() == [[0.0, 2.0]], '厂 ReLu wrong'
    assert operators[1](torch.Tensor([[-1,2]])).tolist() == [[-0.5, 2.0]], '广 LeakyReLu wrong'
    assert operators[2](torch.Tensor([[-1,2]])).tolist() == [[0.2689414322376251, 0.8807970285415649]], '了 Sigmoid wrong'
    assert operators[3](torch.Tensor([[-1,2]])).tolist() == [[-0.7615941762924194, 0.9640275835990906]], '丁 Tanh wrong'
    assert operators[4](torch.Tensor([[-1,2]])).tolist() == [[0.3132616877555847, 2.1269280910491943]], '扎 Softplus wrong'
    assert operators[5](torch.Tensor([[-1,2]])).tolist() == [[-0.8414710164070129, 0.9092974066734314]], '弓 Sin wrong'
    assert operators[6](torch.Tensor([[-1,2]])).tolist() == [[0.5403023362159729, -0.416146844625473]], '引 Cos wrong'
    assert operators[7](torch.Tensor([[-1,2]])).tolist() == [[1.0, 2.0]], '凶 Abs wrong'
    assert operators[8](torch.Tensor([[4, 16]])).tolist() == [[2.0, 4.0]], '风 Sqrt wrong'
    assert operators[9](torch.Tensor([[-1,2], [1,4]])).tolist() == [[0, 6]], '土 SumVerically wrong'
    # assert operators[10](torch.Tensor([[-1,2], [1,4]])).tolist() == [[-1,2], [1,4]], '川 Identity wrong'

def test_normalization_operators():
    operators = hanz.parseHanzLines("""2
    中申 ... 5; 5
    """.split("\n"))
    assert len(operators) == 2
    assert operators[0](torch.randn(20, 5, 35, 45)).shape == torch.Size([20, 5, 35, 45]), '中 InstanceNorm2d wrong'
    assert operators[1](torch.randn(20, 5, 35, 45)).shape == torch.Size([20, 5, 35, 45]), '申 BatchNorm2d wrong'

def test_combine_operators():
    operators = hanz.parseHanzLines("""2
    川川川川川川川川川川
    川川川川正川川川川川
    +艹朋非羽
    """.split("\n"))
    assert len(operators) == 5
    assert operators[0](torch.Tensor([[-1,2]])).tolist() == [[-2, 4]], '+ Add wrong'
    assert operators[1](torch.Tensor([[-1,2]])).tolist() == [[-1,2,-1,2]], '艹 Concat wrong'
    assert operators[2](torch.Tensor([[-1,2]])).tolist() == [[1.0, -2.0], [-2.0, 4.0]], '朋 Matmul wrong'
    assert operators[3](torch.Tensor([[-1,2]])).tolist() == [[5.]], '非 DotProduct wrong'
    assert operators[4](torch.Tensor([[-1,2]])).tolist() == [[1, 4]], '羽 Mul ElementWiseProduct) wrong'

def test_matrix_operators():
    operators = hanz.parseHanzLines("""2
    正昌日吕 ... ; 2 ; -1 ; 0,1
    """.split("\n"))
    assert len(operators) == 4
    assert operators[0](torch.Tensor([[-1,2],[3,4]])).tolist() == [[-1.0, 3.0], [2.0, 4.0]], '正 transpose wrong'
    assert operators[1](torch.Tensor([[-1,2],[3,4]])).shape == torch.Size([2,2]), '昌 LeakyReLu wrong'
    assert operators[2](torch.Tensor([[-1,2],[3,4]])).tolist() == [[3,4]], '日 TakeRow wrong'
    assert operators[3](torch.Tensor([[-1,2],[3,4]])).tolist() == [[-1], [3]], '吕 Split wrong'


def test_grid_operators():
    operators = hanz.parseHanzLines("""3
    田井一 ... 5, kernel_size = (3,3), padding = 1, bias=False; 5, kernel_size = (3,3), padding=1, bias = False ; 100
    """.split("\n"))
    assert len(operators) == 3
    assert operators[0](torch.randn(10, 3, 50, 100)).shape == torch.Size([10, 5, 50, 100]), '田 Conv2d'
    assert operators[1](torch.randn(10, 3, 50, 100)).shape == torch.Size([10, 5, 50, 100]), '井 ConvTranspose2d'
    assert operators[2](torch.ones(2, 3, 10, 10)).shape == torch.Size([2, 300]), '一 Flatten wrong'

def test_other_operators():
    operators = hanz.parseHanzLines("""2
    森亢 ... 5 ; 0.5 ;
    """.split("\n"))
    assert len(operators) == 2
    assert operators[0](torch.Tensor([[-1,2]])).shape == torch.Size([1, 5]), '森 FullyConnected wrong'
    assert operators[1](torch.Tensor([[-1,2],[1,4]])).tolist() == [[0.11920291930437088, 0.11920291930437088], [0.8807970285415649, 0.8807970285415649]], '亢 Softmax wrong'

def test_split_by_parameter_names():
    operators, functions = hanz.parseHanzLines("""pos:2 embedding:4
    |embedding|pos|pos|embedding
    """.split("\n"))
    assert len(functions) == 4
    embedding = [[-1,2,3,4], [5,6,7,8]]
    pos = [[11,12],[-13,-14]]
    embedding_tensor = torch.Tensor(embedding)
    pos_tensor = torch.Tensor(pos)
    # import pdb
    # pdb.set_trace()
    assert functions[0](pos= pos_tensor, embedding= embedding_tensor).tolist() == embedding
    assert functions[1](pos= pos_tensor, embedding= embedding_tensor).tolist() == pos
    assert functions[2](pos= pos_tensor, embedding= embedding_tensor).tolist() == pos
    assert functions[3](pos= pos_tensor, embedding= embedding_tensor).tolist() == embedding

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
    print('----- ', result.shape)
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
    expected_result = [[0.4794255495071411, 0.4794255495071411, 0.4794255495071411, 0.4794255495071411, 0.4794255495071411]]
    expected_result2 = [[0.8775825500488281, 0.8775825500488281]]
    assert result.tolist() == expected_result, 'expecting {} received {}'.format(expected_result, result)
    assert result2.tolist() == expected_result2, 'expecting {} received {}'.format(expected_result2, result2)

def test_repeat_columns():
    modules, functions = hanz.parseHanz('examples/example-decoder.hanz')
    assert len(functions) == 1
    assert len(modules) == 1
    f = functions[0]
    result = f(query_vector = torch.ones(1,2) * 0.5, word_embedding = torch.ones(1,2) * 0.5, positional_encoding = torch.ones(1,2) * 0.5)
    assert result.shape == torch.Size([1,2])
