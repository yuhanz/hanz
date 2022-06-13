import torch.nn as nn
import torch
import re
from functools import partial
import pdb

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
    def __init__(self, fn):     # Let the fn be a torch function. for example: torch.sin
        super().__init__()
        self.fn = fn
    def forward(self, x):
        r = self.fn(x)
        return r


class Add(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return x+ self.model(x)

f = open("./example.hanz","r")
lines = f.readlines()
print(lines)

dim = int(lines[0])
lines.pop(0)
modules = [None]

for line in lines:
  [operators, configs] = (line.split("...") + [''])[0:2]
  operators = operators.strip()
  if len(operators) == 0:
      continue
  config_list = configs.strip().split(";")
  config_list = config_list + [''] * max(len(operators) - len(config_list), 0)
  config_list = list(map(lambda s: s.strip(), config_list))

  new_modules = []
  # print(config_list)
  # print(operators)
  z = zip(operators, config_list)
  commands = list(z)
  # for operator, config in zip(operators, config_list):
  for operator, config in commands:
      # pdb.set_trace()
      m = modules.pop(0)
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
          m = m + m2
      elif operator == '艹':
          m2 = modules.pop(0)
          m = torch.concat((m, m2), 0)
      else:
        raise Exception('Unrecognized operator: {}'.format(operator))
      if m:
        m = nn.Sequential(m, new_module)
      else:
        m = new_module
      # pdb.set_trace()
      new_modules.append(m)
  modules = new_modules

print(modules)
print(modules[0](torch.Tensor([1,2])))
