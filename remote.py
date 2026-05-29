import numpy
import tvm_ffi

import tvm
from tvm import rpc, te, ir, relax
from tvm.contrib import utils
from tvm.script import ir as I, relax as R

n = tvm.runtime.convert(1024)
A = te.placeholder((n,), name="A")
B = te.compute((n,), lambda i: A[i] + 1.0, name="B")
func = te.create_prim_func([A, B]).with_attr("global_symbol", "add_one")
mod = ir.IRModule.from_expr(func)

target = tvm.target.Target({
    "kind": "llvm",
    "device": "arm_cpu",
    "mtriple": "armv7l-linux-gnueabihf",
    "mcpu": "cortex-a9",
    "mattr": ["+neon"],
    "mfloat-abi": "hard",
})

ex = tvm.compile(mod, target=target)
# save the lib at a local temp folder
temp = utils.tempdir()
path = temp.relpath("lib.tar")
ex.export_library(path)
host = "10.100.4.202"
port = 9090
remote = rpc.connect(host, port)

remote.upload(path)
func = remote.load_module("lib.tar")

# create arrays on the remote device
dev = remote.cpu()
a = tvm.runtime.tensor(numpy.random.uniform(size=1024).astype(A.dtype), dev)
b = tvm.runtime.tensor(numpy.zeros(1024, dtype=A.dtype), dev)
# the function will run on the remote device
func(a, b)
numpy.testing.assert_equal(b.numpy(), a.numpy() + 1)

time_f = func.time_evaluator("add_one", dev, number=10)
cost = time_f(a, b).mean
print(f"{cost:g} secs/op")

################################################################################

import tvm.relax.backend.contrib.strela  # registers patterns
patterns = relax.backend.pattern_registry.get_patterns_with_prefix("strela")

@I.ir_module
class MatmulReLU:
    @R.function
    def main(
        x: R.Tensor((2, 4), "int32"),
        w: R.Tensor((4, 8), "int32"),
    ) -> R.Tensor((2, 8), "int32"):
        with R.dataflow():
            y = R.matmul(x, w)
            z = R.nn.relu(y)
            R.output(z)
        return z

mod = MatmulReLU
mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod)
mod = relax.transform.MergeCompositeFunctions()(mod)
mod = relax.transform.RunCodegen()(mod)
mod.show()

ex = tvm.compile(mod, target=target)
temp = utils.tempdir()
path = temp.relpath("lib_strela.tar")
ex.export_library(path)
remote.upload(path)
ex = remote.load_module("lib_strela.tar")
vm = relax.VirtualMachine(ex, dev)

x = tvm.runtime.tensor(numpy.ones((2, 4), dtype="int32"), dev)
xn = tvm.runtime.tensor(-numpy.ones((2, 4), dtype="int32"), dev)
w = tvm.runtime.tensor(numpy.ones((4, 8), dtype="int32"), dev)
z = vm["main"](x, w)
print(z)
z = vm["main"](xn, w)
print(z)
