import torch
import hanz
import sys

file_name = sys.argv[1]

modules = hanz.parseHanz(file_name)

print(modules)
print(modules[0](torch.Tensor([1,2])))
