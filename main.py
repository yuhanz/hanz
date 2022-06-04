import torch.nn as nn
import torch
import re

class Custom(nn.Module):
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
m = None

for line in lines:
  operator = line[0]
  value_1 = float(re.sub(r"\s", "", line[1:].replace("...", "")) or "0")
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
  else:
    raise Exception('Unrecognized operator: {}'.format(operator))
  if m:
    m = nn.Sequential(m, new_module)
  else:
    m = new_module

print(m)
print(m(torch.Tensor([1,2])))
