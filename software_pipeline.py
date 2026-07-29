import tvm
from tvm import s_tir
from tvm.script import tirx as T, ir as I

@I.ir_module
class MyModule:
    @T.prim_func
    def main(A: T.Buffer((1024,), "float32"), C: T.Buffer((1024,), "float32")):
        # Allocate a local buffer to hold intermediate data
        B = T.alloc_buffer((1,), "float32", scope="local")

        for i in T.serial(1024):
            with T.sblock("read"):
                T.reads(A[i])
                T.writes(B[0])
                B[0] = A[i]

            with T.sblock("compute"):
                T.reads(B[0])
                T.writes(C[i])
                C[i] = B[0] * 2.0

# Create a schedule from the module
sch = s_tir.Schedule(MyModule)

# Retrieve the target loop (the `i` loop surrounding our blocks)
block_read = sch.get_sblock("read")
loop_i = sch.get_loops(block_read)[0]

# 1. Assign Stages
# The loop body has two statements: the "read" block and the "compute" block.
# We put "read" in Stage 0, and "compute" in Stage 1.
sch.annotate(loop_i, "software_pipeline_stage", [0, 1])

# 2. Assign Order
# In the steady-state pipeline body, we want the "compute" (Stage 1) to execute
# before the "read" (Stage 0) of the *next* iteration to minimize register/buffer collisions.
sch.annotate(loop_i, "software_pipeline_order", [1, 0])

mod = sch.mod
mod.show()
mod = s_tir.transform.InjectSoftwarePipeline()(sch.mod)
mod.show()
