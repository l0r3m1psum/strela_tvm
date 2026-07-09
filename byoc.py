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
        x: R.Tensor((2, 4), "int32"),
        w: R.Tensor((4, 8), "int32"),
    ) -> R.Tensor((2, 8), "int32"):
        with R.dataflow():
            y = R.matmul(x, w)
            z = R.nn.relu(y)
            R.output(z)
        return z

if False:
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
    x_np = numpy.random.randn(2, 4).astype("int32")
    w_np = numpy.random.randn(4, 8).astype("int32")

    with tvm.transform.PassContext(opt_level=3):
        build = relax.build(mod, target)

    vm = relax.VirtualMachine(build, tvm.cpu())
    result = vm["main"](tvm.runtime.tensor(x_np, tvm.cpu()), tvm.runtime.tensor(w_np, tvm.cpu()))

################################################################################

import tvm.relax.backend.contrib.strela

patterns = relax.backend.pattern_registry.get_patterns_with_prefix("strela")
print("Registered patterns:", [p.name for p in patterns])

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

with tvm.transform.PassContext(opt_level=3):
    build = relax.build(mod, target)

vm = relax.VirtualMachine(build, tvm.cpu())

numpy.random.seed(0)
x_np = numpy.random.randn(2, 4).astype("int32")
w_np = numpy.random.randn(4, 8).astype("int32")

# result = vm["main"](tvm.runtime.tensor(x_np, tvm.cpu()), tvm.runtime.tensor(w_np, tvm.cpu()))
# print(result)

################################################################################

@I.ir_module
class CenteredBilinearProduct:
    @R.function
    def main(
        x: R.Tensor((2, 4), "int8"),
        w: R.Tensor((4, 8), "int8"),
        b: R.Tensor((8,), "int32"),
    ):
        with R.dataflow():
            y = R.matmul(
                x.astype("int32") - R.const(12, "int8").astype("int32"),
                w.astype("int32") - R.const(34, "int8").astype("int32"),
            ) + b
            R.output(y)
        return y

mod = CenteredBilinearProduct
mod = relax.transform.Normalize()(mod)
mod = relax.transform.CanonicalizeBindings()(mod)
mod.show()
mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod)
mod.show()
mod = relax.transform.MergeCompositeFunctions()(mod)
print("After partitioning:")
mod.show()
mod = relax.transform.RunCodegen()(mod)
print("After codegen:")
mod.show()

@relax.expr_functor.visitor
class DataflowToDotVisitor(relax.PyExprVisitor):
    def __init__(self):
        super().__init__()
        self._nodes = []
        self._edges = []
        self._op_count = 0
        self._const_count = 0
        self._in_dataflow_block = False

    def generate_dot(self, expr: relax.Expr) -> str:
        """
        Visits the IR and returns a DOT graph string representation.
        """
        # Reset state for fresh generation
        self._nodes = []
        self._edges = []
        self._op_count = 0
        self._const_count = 0

        self.visit_expr(expr)

        # Assemble the DOT string
        lines = [
            "digraph RelaxDataflow {",
            '  rankdir="TB";',
            '  node [fontname="Helvetica"];',
            '  edge [fontname="Helvetica", fontsize=10];'
        ]
        lines.extend(self._nodes)
        lines.extend(self._edges)
        lines.append("}")
        return "\n".join(lines)

    def visit_dataflow_block_(self, block: relax.DataflowBlock) -> None:
        assert not self._in_dataflow_block, "Nested data-flow block?!"
        self._in_dataflow_block = True

        self._nodes.append("  subgraph cluster_dataflow {")
        self._nodes.append('    label="Dataflow Block";')
        self._nodes.append('    style=dashed;')
        self._nodes.append('    color=grey;')

        # Visit all bindings in the block
        super().visit_dataflow_block_(block)

        self._nodes.append("  }")
        self._in_dataflow_block = False

    def visit_var_binding_(self, binding: relax.VarBinding) -> None:
        if self._in_dataflow_block:
            var_name = binding.var.name_hint
            self._declare_var_node(binding.var, var_name)
            self._process_value(binding.value, var_name)
        else:
            return super().visit_var_binding_(binding)

    def visit_match_cast_(self, binding: relax.MatchCast) -> None:
        if self._in_dataflow_block:
            var_name = binding.var.name_hint
            self._declare_var_node(binding.var, var_name)

            # Represent the MatchCast as an operation
            self._op_count += 1
            op_id = f"op_match_{self._op_count}"
            self._nodes.append(f'    "{op_id}" [label="MatchCast", shape=ellipse, style=filled, fillcolor=lightyellow];')
            self._edges.append(f'  "{op_id}" -> "{var_name}";')
            self._process_arg(binding.value, op_id)
        else:
            return super().visit_match_cast_(binding)

    # --- Helper Methods ---

    def _declare_var_node(self, var: relax.Var, var_name: str):
        """Declares the shape and color of the variable node."""
        if isinstance(var, relax.DataflowVar):
            self._nodes.append(f'    "{var_name}" [label="{var_name}", shape=box, style=filled, fillcolor=lightblue];')
        else:
            # Standard var, usually an output bound out of the dataflow block
            self._nodes.append(f'    "{var_name}" [label="{var_name}", shape=box, style=filled, fillcolor=lightgreen];')

    def _process_value(self, val: relax.Expr, target_var: str):
        """Processes the bound value and links it to the target variable."""
        if isinstance(val, relax.Call):
            self._op_count += 1
            op_id = f"op_{self._op_count}"

            # Extract operation name (e.g., relax.add, relax.nn.conv2d)
            if hasattr(val.op, "name"):
                op_name = val.op.name
            else:
                op_name = str(val.op)

            self._nodes.append(f'    "{op_id}" [label="{op_name}", shape=ellipse];')
            self._edges.append(f'  "{op_id}" -> "{target_var}";')

            for arg in val.args:
                self._process_arg(arg, op_id)

        elif isinstance(val, relax.TupleGetItem):
            self._op_count += 1
            op_id = f"op_tgi_{self._op_count}"
            self._nodes.append(f'    "{op_id}" [label="TupleGetItem({val.index})", shape=ellipse];')
            self._edges.append(f'  "{op_id}" -> "{target_var}";')
            self._process_arg(val.tuple_value, op_id)

        elif isinstance(val, relax.Tuple):
            self._op_count += 1
            op_id = f"op_tuple_{self._op_count}"
            self._nodes.append(f'    "{op_id}" [label="Tuple", shape=ellipse];')
            self._edges.append(f'  "{op_id}" -> "{target_var}";')
            for arg in val.fields:
                self._process_arg(arg, op_id)

        else:
            # Fallback for constants or directly aliased variables
            self._op_count += 1
            op_id = f"op_generic_{self._op_count}"
            self._nodes.append(f'    "{op_id}" [label="{type(val).__name__}", shape=ellipse];')
            self._edges.append(f'  "{op_id}" -> "{target_var}";')
            self._process_arg(val, op_id)

    def _process_arg(self, arg: relax.Expr, target_id: str):
        """Draws edges from inputs to operations."""
        if isinstance(arg, relax.Var):
            # Graphviz will auto-create this node if it's an external input missing from the block
            self._edges.append(f'  "{arg.name_hint}" -> "{target_id}";')
        elif isinstance(arg, relax.Constant):
            self._const_count += 1
            const_id = f"const_{self._const_count}"
            self._nodes.append(f'    "{const_id}" [label="Const", shape=diamond, style=filled, fillcolor=lightgrey];')
            self._edges.append(f'  "{const_id}" -> "{target_id}";')
        elif isinstance(arg, relax.ShapeExpr):
            self._const_count += 1
            shape_id = f"shape_{self._const_count}"
            self._nodes.append(f'    "{shape_id}" [label="Shape{list(arg.values)}", shape=diamond, style=filled, fillcolor=lightgrey];')
            self._edges.append(f'  "{shape_id}" -> "{target_id}";')

@I.ir_module
class Overlap:
    @R.function
    def main(
        x: R.Tensor(("N", 16), dtype="int8"),
        w: R.Tensor((16, 16), dtype="int8"),
    ):
        with R.dataflow():
            zp = R.const(0, dtype="int8").astype("int32")
            y = R.matmul(x.astype("int32")-zp, w.astype("int32")-zp).astype("int8")
            # if y is changed with x it matches correctly...
            y2 = R.matmul(y.astype("int32")-zp, w.astype("int32")-zp).astype("int8")
            R.output(y2)
        return y2

mod = Overlap
mod = relax.transform.Normalize()(mod)
mod = relax.transform.CanonicalizeBindings()(mod)
mod.show()
mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod)
mod.show()
mod = relax.transform.MergeCompositeFunctions()(mod)
print("After partitioning:")
mod.show()
mod = relax.transform.RunCodegen()(mod)
print("After codegen:")
mod.show()

raise SystemExit(0)

@I.ir_module
class MinimalCyclicModule:
    @R.function
    def main(
        x: R.Tensor((1, 16), dtype="int8"),
        w1: R.Tensor((16, 16), dtype="int8"),
        w2: R.Tensor((16, 16), dtype="int8")
    ) -> R.Tensor((1, 16), dtype="int32"):
        with R.dataflow():
            # 1. The Shared Internal Node
            # This constant cast is matched as an internal node by BOTH pattern matches.
            zp_casted = R.astype(R.const(0, "int8"), dtype="int32")

            # 2. First Pattern Match
            x_casted = R.astype(x, dtype="int32")
            x_centered = R.subtract(x_casted, zp_casted)

            w1_casted = R.astype(w1, dtype="int32")
            w1_centered = R.subtract(w1_casted, zp_casted)

            matmul1 = R.matmul(x_centered, w1_centered, out_dtype="void")

            # 3. Intermediate nodes OUTSIDE the pattern
            # These nodes break the adjacency between matmul1 and matmul2
            intermediate_int32 = R.add(matmul1, R.const(1, dtype="int32"))
            intermediate_int8 = R.astype(intermediate_int32, dtype="int8")

            # 4. Second Pattern Match
            inter_casted = R.astype(intermediate_int8, dtype="int32")
            inter_centered = R.subtract(inter_casted, zp_casted) # <-- Reusing the shared node!

            w2_casted = R.astype(w2, dtype="int32")
            w2_centered = R.subtract(w2_casted, zp_casted)

            matmul2 = R.matmul(inter_centered, w2_centered, out_dtype="void")

            R.output(matmul2)
        return matmul2

mod = MinimalCyclicModule

visitor = DataflowToDotVisitor()
dot_string = visitor.generate_dot(mod["main"])
print(dot_string)

mod = relax.transform.Normalize()(mod)
mod = relax.transform.CanonicalizeBindings()(mod)
mod.show()
mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod)
mod.show()
mod = relax.transform.MergeCompositeFunctions()(mod)
print("After partitioning:")
mod.show()
mod = relax.transform.RunCodegen()(mod)
print("After codegen:")
mod.show()
