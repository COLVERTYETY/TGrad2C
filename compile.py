from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
import os
from tinygrad.jit import TinyJit
from model import Model
import argparse
import struct
import ctypes

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True, help="name of the model to compile")
args = parser.parse_args()
modelname = args.name


myModel = Model()
myModel.load(modelname+".npz")
# print(myModel)

@TinyJit
def run(x):
    return myModel(x).realize()

the_input = Tensor.randn(1,requires_grad=False)+1 -1 # +1 -1 so that remains a buffer for jit
# twice to run the JIT
the_output = run(the_input)
the_output = run(the_input)
print("model jitted successfully")

def compile_net(run, special_names):
  # functions that run the net
  functions = {}
  bufs = {}
  bufnum = 0
  statements = []
  bufs_to_save = {}
  for fxn,args in run.jit_cache:
    print("fxn", fxn.name, "args", args)
    functions[fxn.name] = fxn.prg   # NOTE: this assumes all with the same name are the same
    cargs = []
    for i,arg in enumerate(args):
      print("arg", arg, i)
      key = id(arg)
      if key not in bufs:
        if key in special_names:
          bufs[key] = (special_names[key], len(arg._buf))
        else:
          bufs[key] = (f"buf_{bufnum}", len(arg._buf))
          bufnum += 1
          if i > 0: bufs_to_save[bufs[key][0]] = arg   # if first usage of a buffer is not an the_output, and it's not a special name
      cargs.append(bufs[key][0])
    statements.append(f"{fxn.name}({', '.join(cargs)});")

  return functions, statements, bufs, bufs_to_save

# hack to put the inputs back
assert len(run.input_replace) == 1, f"didn't get one input to replace {run.input_replace}"
for (j,i),idx in run.input_replace.items():
  run.jit_cache[j][1][i] = the_input.lazydata.realized
# TODO: fetch this from the jit in self.input_replace and self.ret (hint: use get_parameters on self.ret)
special_names = {id(the_input.lazydata.realized): "the_input", id(the_output.lazydata.realized): "the_output"}

# print("special_names", special_names)
N_kernels = len(run.jit_cache)
print("starting codegen for model", modelname, "with", N_kernels, "kernels")
functions, statements, bufs, bufs_to_save = compile_net(run, special_names)
print("functions", functions)
print("statements", statements)
print("bufs", bufs)
print("bufs to save",bufs_to_save)


# c header
cprog = [ "#include <math.h>", "#define max(x,y) ((x>y)?x:y)", f"// OPT={os.environ['OPT']} N_kernels={N_kernels}" ]

# save the weights
for name,cl in bufs_to_save.items():
    weight = ','.join([str( x) for x in cl._buf])
    print("weight", weight)
    cprog.append(f"float {name}_data[] = {{{weight}}};")

# buffers (empty + weights)
cprog += [f"float {name}[{len}];" if name not in bufs_to_save else f"float *{name} = (float *){name}_data;" for name,len in bufs.values()]

# the functions
cprog += list(functions.values())

# the net
cprog += ["void net() {"] + statements + ["}"]

cprog += ["float forward(float input){ the_input[0] = input; net(); return the_output[0]; }"]

# save the c program
save_name = f"{modelname}_{os.environ['OPT']}.c"
with open(save_name, "w") as f:
    f.write("\n".join(cprog))


print("CODEGEN SUCCESSFULLY")
print(f"| save_name:{save_name}")