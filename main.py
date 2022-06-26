import torch.nn as nn
import torch
import re
from functools import partial
import pdb
import sys

class Custom(nn.Module):
    def __init__(self, fn, no_argument = False, name = None):     # Let the fn be a torch function. for example: torch.sin
        super().__init__()
        self.fn = fn
        self.no_argument = no_argument
        self.name = name or self.__class__.__name__
    def forward(self, x):
        if self.no_argument:
          return self.fn()
        r = self.fn(x)
        return r
    def _get_name(self):
        return self.name

class CustomCombine(nn.Module):
    def __init__(self, fn, fn1, fn2, name = None):     # Let the fn be a torch function. for example: torch.sin
        super().__init__()
        self.fn = fn
        self.fn1 = fn1
        self.fn2 = fn2
        self.name = name or self.__class__.__name__
    def forward(self, x):
        r1 = self.fn1(x)
        r2 = self.fn2(x)
        r = self.fn(r1, r2)
        return r
    def _get_name(self):
        return self.name


class Add(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return x+ self.model(x)

file_name = sys.argv[1]
f = open(file_name, "r")
lines = f.readlines()
print(lines)

dim = int(lines[0])
lines.pop(0)
modules = [None]
dims = [dim]

line_number = 1
line_content = None

def parseLine(line):
  line = re.sub(r"#.*", "", line.strip())
  [operators, configs] = (line.split("...") + [''])[0:2]
  operators = operators.strip()
  config_list = configs.strip().split(";")
  config_list = config_list + [''] * max(len(operators) - len(config_list), 0)
  config_list = list(map(lambda s: s.strip(), config_list))
  return [operators, config_list]

def parseOneFloat(config):
  value_1 = float(re.sub(r"\s", "", config) or "0")
  return value_1

# Return (module, output_dim)
def interpretModule(operator, config, dim):
  output_dim = dim
  if operator == '森':
    output_dim = int(parseOneFloat(config))
    new_module = nn.Linear(dim, output_dim)  # TODO: bias=
  elif operator == '厂':
    new_module = nn.ReLU()
  elif operator == '广':
    new_module = nn.LeakyReLU(parseOneFloat(config))
  elif operator == '了':
    new_module = nn.Sigmoid()
  elif operator == '丁':
    new_module = nn.Tanh()
  elif operator == '亢':
    d = int(parseOneFloat(config))
    new_module = nn.Softmax(dim=max(0, d))
  elif operator == '扎':
    new_module = nn.Softplus()
  elif operator == '弓':
    new_module = Custom(torch.sin, name = 'Sin')
  elif operator == '引':
    new_module = Custom(torch.cos, name = 'Cos')
  elif operator == '凶':
    new_module = Custom(torch.abs, name = 'Abs')
  elif operator == '吕':
    values = list(map(lambda s: float(s.strip()), (re.sub(r"\s", "", config).split(','))))
    assert len(values) == 2, "Expecting 2 numerical values after 吕"
    new_module = Custom(lambda x: x[:, values[0]:values[1]], name = 'SelectColumns')
  elif operator == '目':
    # TODO: parse 3 numerical values
    new_module = Custom(partial(torch.linspace, start, end, steps), no_argument = True, name = 'Linspace')
  else:
    return (None, output_dim)
  return (new_module, output_dim)

try:
    for line in lines:
      line_number = line_number + 1
      line_content = line = line.strip()
      operators, config_list = parseLine(line)

      new_modules = []
      new_dims = []
      z = zip(operators, config_list)
      commands = list(z)
      # for operator, config in zip(operators, config_list):
      num_commands = len(commands)
      num_modules = len(modules)
      if num_commands > num_modules:
          for i in range(0, num_commands - num_modules):
              modules.append(modules[-1])
              dims.append(dims[-1])
      for operator, config in commands:
          m = modules.pop(0)
          dim = dims.pop(0)
          new_module = None
          if operator == '川':
            new_module = m
          else:
            new_module, output_dim = interpretModule(operator, config, dim)

          # if line_number == 9:
          #   pdb.set_trace()
          if m == None or operator == '川':
            m = new_module
          elif new_module != None:
            m = nn.Sequential(m, new_module)
          else:
            if operator == '+':
              m2 = modules.pop(0)
              new_module = CustomCombine(torch.add, m, m2, name = 'Add')
              #pdb.set_trace()
            elif operator == '艹':
              m2 = modules.pop(0)
              new_module = CustomCombine(torch.concat, m, m2, name = 'Concat')
            else:
              raise Exception('Unrecognized operator: {}'.format(operator))
            m = new_module
          new_modules.append(m)
          new_dims.append(dim)
      modules = new_modules
      dims = new_dims
except Exception as ex:
  print("Exception happened processing file {} at\n  line #{}: {}\n           {}".format(file_name, line_number, line_content, ex))
  raise ex

print(modules)
#print(modules[0](torch.Tensor([1,2])))
