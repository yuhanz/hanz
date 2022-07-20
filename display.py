import hanz
import torch.nn as nn
import torch
import sys

file_name = sys.argv[1]
modules = hanz.parseHanz(file_name)
print(modules)
