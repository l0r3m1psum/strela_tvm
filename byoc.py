import numpy

import tvm
import tvm.relax.backend.contrib.example_npu  # registers patterns
from tvm import relax
from tvm.script import relax as R, ir as I

target = tvm.target.Target("llvm")

patterns = relax.backend.pattern_registry.get_patterns_with_prefix("example_npu")
print("Registered patterns:", [p.name for p in patterns])

@I.ir_module
class MatmulReLU:
    @R.function
    def main(
        x: R.Tensor((2, 4), "float32"),
        w: R.Tensor((4, 8), "float32"),
    ) -> R.Tensor((2, 8), "float32"):
        with R.dataflow():
            y = R.matmul(x, w)
            z = R.nn.relu(y)
            R.output(z)
        return z

mod = MatmulReLU
mod.show()
mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod)
mod.show()
mod = relax.transform.MergeCompositeFunctions()(mod)
print("After partitioning:")
mod.show()
mod = relax.transform.RunCodegen()(mod)
print("After codegen:")
mod.show()

numpy.random.seed(0)
x_np = numpy.random.randn(2, 4).astype("float32")
w_np = numpy.random.randn(4, 8).astype("float32")

with tvm.transform.PassContext(opt_level=3):
    build = relax.build(mod, target)

vm = relax.VirtualMachine(build, tvm.cpu())
result = vm["main"](tvm.runtime.tensor(x_np, tvm.cpu()), tvm.runtime.tensor(w_np, tvm.cpu()))
