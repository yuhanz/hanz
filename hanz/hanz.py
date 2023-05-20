import torch.nn as nn
import torch
import re
from functools import partial
from ast import literal_eval as make_tuple
# import pdb

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


def removeComments(lines):
  return list(map(lambda line: re.sub(r"#.*", "", line).strip(), lines))


def parseLine(line):
  line = re.sub(r"#.*", "", line.strip())
  if(line.strip() == ''):
      return [[], []]
  [operators, configs] = (line.split("...") + [''])[0:2]
  operators = operators.strip()
  config_list = configs.strip().split(";")
  if len(config_list) == 1 and len(list(filter(lambda o: o != '川', operators))) == 1:
      return [operators, list(map(lambda o: config_list[0] if o != '川' else None, operators))]
  config_list = config_list + [''] * max(len(operators) - len(config_list), 0)
  config_list = list(map(lambda s: s.strip(), config_list))
  return [operators, config_list]

def parseOneFloat(config):
  config = config.split(',',1)[0]
  value_1 = float(re.sub(r"\s", "", config) or "0")
  return value_1

def parseInts(config):
  return list(map(lambda s: int(s.strip()), (re.sub(r"\s", "", config).split(','))))

def parseForNamedParams(config):
  parts = re.split(r',\s*(?![^()]*\))', config)
  named_params = dict(list(map(lambda pp: map(lambda ppp: ppp.strip(), pp.split("=",1)), (filter(lambda p: "=" in p, parts)))))
  return named_params

# Return (module, output_dim)
def interpretModule(operator, config, dim):
  output_dim = dim
  named_params = parseForNamedParams(config)
  params = dict([[k, make_tuple(v) if k != 'padding_mode' else v] for k,v in named_params.items()])
  if operator == '森':
    output_dim = int(parseOneFloat(config))
    assert output_dim > 0, "Expecting 1 positive int value after 森"
    new_module = nn.Linear(dim, output_dim, **params)
  elif operator == '厂':
    new_module = nn.ReLU(**params)
  elif operator == '广':
    new_module = nn.LeakyReLU(parseOneFloat(config), **params)
  elif operator == '中':
    new_module = nn.InstanceNorm2d(parseOneFloat(config), **params)
  elif operator == '申':
    new_module = nn.BatchNorm2d(parseOneFloat(config), **params)
  elif operator == '了':
    new_module = nn.Sigmoid()
  elif operator == '丁':
    new_module = nn.Tanh()
  elif operator == '亢':
    d = int(parseOneFloat(config))
    new_module = nn.Softmax(dim=max(0, d))
  elif operator == '扎':
    new_module = nn.Softplus(**params)
  elif operator == '弓':
    new_module = Custom(torch.sin, name = 'Sin')
  elif operator == '引':
    new_module = Custom(torch.cos, name = 'Cos')
  elif operator == '凶':
    new_module = Custom(torch.abs, name = 'Abs')
  elif operator == '风':
    new_module = Custom(torch.sqrt, name = 'Sqrt')
  elif operator == '正':
    new_module = Custom(torch.transpose, name = 'Transpose') # TODO
  elif operator == '一':
    new_module = nn.Flatten(**params)
    output_dim = int(parseOneFloat(config))
  elif operator == '吕':
    values = parseInts(config)
    assert len(values) == 2, "Expecting 2 numerical values after 吕"
    new_module = Custom(lambda x: x[:, values[0]:values[1]], name = 'SelectColumns')
    output_dim = values[1] - values[0]
  elif operator == '田':
    output_dim = int(parseOneFloat(config))
    new_module = nn.Conv2d(dim, output_dim, **params)
  elif operator == '井':
    output_dim = int(parseOneFloat(config))
    new_module = nn.ConvTranspose2d(dim, output_dim, **params)
  else:
    return (None, output_dim)
  return (new_module, output_dim)

def moduleListToModuleFn(l):
  return nn.Sequential(*l) if len(l) >1 else l[0]

def combineModuleLists(operator, module_list, dim, module_lists, dims):
  m2_list = module_lists.pop(0)
  dim2 = dims.pop(0)
  output_dim = dim
  if operator == '+' or operator == '十':
    new_moduleX = CustomCombine(torch.add, nn.Sequential(*module_list), nn.Sequential(*m2_list), name = 'Add')
  elif operator == '艹':
    new_moduleX = CustomCombine(lambda x1, x2: torch.cat((x1, x2), 1), nn.Sequential(*module_list), nn.Sequential(*m2_list), name = 'Concat')
    output_dim = dim + dim2
  elif operator == '昌':
    new_moduleX = CustomCombine(torch.matmul, nn.Sequential(*module_list), nn.Sequential(*m2_list), name = 'MatMultiply')
  else:
    return [None, output_dim]
  return [[new_moduleX], output_dim]

def parseHanz(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    return parseHanzLines(lines, file_name)

def parseHanzLines(lines, file_name = None):
    lines = removeComments(lines)
    dim = int(lines[0])
    lines.pop(0)
    module_lists = [[]]
    dims = [dim]

    line_number = 1
    line_content = None

    try:
        for line in lines:
          line_number = line_number + 1
          line_content = line = line.strip()
          operators, config_list = parseLine(line)
          if(len(operators) == 0):
            continue
          # if line_number == 3:
          #     pdb.set_trace()

          new_module_lists = []
          new_dims = []
          z = zip(operators, config_list)
          commands = list(z)
          # for operator, config in zip(operators, config_list):
          num_commands = len(commands)
          num_modules = len(module_lists)
          if num_commands > num_modules:
              for i in range(0, num_commands - num_modules):
                  module_lists.append(module_lists[-1].copy())
                  dims.append(dims[-1])
          for operator, config in commands:
              m_list = module_lists.pop(0)
              dim = dims.pop(0)
              output_dim = dim
              new_module = None
              if operator != '川':
                new_module, output_dim = interpretModule(operator, config, dim)

              if m_list == None:
                m_list = [new_module]
              elif operator == '川':
                pass
              elif new_module != None:
                m_list.append(new_module)
              else:
                m_list, output_dim = combineModuleLists(operator, m_list, dim, module_lists, dims)
                if m_list == None:
                  raise Exception('Unrecognized operator: {}'.format(operator))
              new_module_lists.append(m_list)
              new_dims.append(output_dim)
          module_lists = new_module_lists
          dims = new_dims
    except Exception as ex:
      print("Exception happened processing file {} at\n  line #{}: {}\n           {}".format(file_name, line_number, line_content, ex))
      raise ex

    return list(map(moduleListToModuleFn, module_lists))
