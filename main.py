import torch.nn as nn
import torch
import re
from functools import partial
import pdb
import sys

class Custom(nn.Module):
    def __init__(self, fn, no_argument = False):     # Let the fn be a torch function. for example: torch.sin
        super().__init__()
        self.fn = fn
        self.no_argument = no_argument
    def forward(self, x):
        if self.no_argument:
          return self.fn()
        r = self.fn(x)
        return r

class CustomCombine(nn.Module):
    def __init__(self, fn, fn1, fn2):     # Let the fn be a torch function. for example: torch.sin
        super().__init__()
        self.fn = fn
        self.fn1 = fn1
        self.fn2 = fn2
    def forward(self, x):
        r1 = self.fn1(x)
        r2 = self.fn2(x)
        r = self.fn(r1, r2)
        return r


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

for line in lines:
  [operators, configs] = (line.split("...") + [''])[0:2]
  operators = operators.strip()
  if len(operators) == 0:
      continue
  config_list = configs.strip().split(";")
  config_list = config_list + [''] * max(len(operators) - len(config_list), 0)
  config_list = list(map(lambda s: s.strip(), config_list))

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
      # pdb.set_trace()
      m = modules.pop(0)
      dim = dims.pop(0)
      value_1 = float(re.sub(r"\s", "", config) or "0")
      if operator == '森':
        output_dim = int(value_1)
        new_module = nn.Linear(dim, output_dim)  # TODO: bias=
        dim = output_dim
      elif operator == '厂':
        new_module = nn.ReLU()
      elif operator == '广':
        new_module = nn.LeakyReLU(value_1)
      elif operator == '了':
        new_module = nn.Sigmoid()
      elif operator == '丁':
        new_module = nn.Tanh()
      elif operator == '亢':
        d = int(value_1)
        new_module = nn.Softmax(dim=max(0, d))
      elif operator == '扎':
        new_module = nn.Softplus()
      elif operator == '川':
        new_module = m
      elif operator == '弓':
        new_module = Custom(torch.sin)
      elif operator == '引':
        new_module = Custom(torch.cos)
      elif operator == '目':
        # TODO: parse 3 numerical values
        new_module = Custom(partial(torch.linspace, start, end, steps), no_argument = True)
      elif operator == '+':
          m2 = modules.pop(0)
          m = CustomCombine(torch.add, m, m2)
      elif operator == '艹':
          m2 = modules.pop(0)
          m = torch.concat((m, m2), 0)
      else:
        raise Exception('Unrecognized operator: {}'.format(operator))
      if m and operator != '川':
        m = nn.Sequential(m, new_module)
      else:
        m = new_module
      # pdb.set_trace()
      new_modules.append(m)
      new_dims.append(dim)
  modules = new_modules
  dims = new_dims

print(modules)
print(modules[0](torch.Tensor([1,2])))
