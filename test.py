import hanz

def pipelineToString(pipeline):
    return ",".join(map(lambda m: m._get_name(), pipeline))

def test_example():
    modules = hanz.parseHanz('examples/example.hanz')
    assert len(modules) == 1
    assert pipelineToString(modules[0]) == "Linear,LeakyReLU,ReLU,Sigmoid,Softplus,Softmax,Sin"

def test_multiple_module_pipeliness():
    modules = hanz.parseHanz('examples/example2a.hanz')
    assert len(modules) == 2
    assert pipelineToString(modules[0]) == "Sin,Linear,ReLU"
    assert pipelineToString(modules[1]) == "Linear,LeakyReLU,ReLU,Sigmoid,Softplus,Softmax,Sin"

def test_combine_multiple_module_pipes():
    modules = hanz.parseHanz('examples/example2.hanz')
    assert len(modules) == 1
    assert isinstance(modules[0], hanz.CustomCombine)
    assert pipelineToString(modules[0].fn1) == "Sin,Linear,ReLU"
    assert pipelineToString(modules[0].fn2) == "Linear,LeakyReLU,ReLU,Sigmoid,Softplus,Softmax,Sin"e
