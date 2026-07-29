"""As with many things neural networks the tflite format is a mess... It is
based on FlatBuffers this creates some interesting headaches given the fact that
the format aims to be both forward and backward compatible. In particular the
tflite.Model.Version() will pretty much always return 3 because previous
versions (i.e. 0, 1 and 2 should all be beta versions) and versions after the 3
need to be discriminated looking at the operator versions (i.e. if an operator
with a certain version that we know was not available until version 3c we know
that the model version is at least that).

On the topic of operator versions if we are reading a tflite file with a certain
version, say 3a, using a version of the tflite module that has a been compiled
with newer schema, say 3d, and we query for an operator option say
QuantizedBiasType of CONV_2D we are going to get the default, which is
tflite.TensorType.FLOAT32 i.e. 0. But we cannot take it at face value since we
first have to check that the operator version is at least 8. In this particular
case the default is the worst possible since it should have been
tflite.TensorType.INT32.

AFAWK there is no automated way to check this kind of version inconsistencies.
The only way is reading the comments in the FlatBuffers schema and adding the
checks to your code.

TFLite Flatbuffers schema
https://github.com/tensorflow/tensorflow/blob/0e79e851ea0f040c1dd5092cf135137af5e8b4af/tensorflow/compiler/mlir/lite/schema/schema.fbs

To add insult to the injury
[LiteRT runtime uses mixed rounding modes across ops](https://github.com/google-ai-edge/LiteRT/issues/7441).
This basically means that given an operation, say CONV2D, based on the available
delegates e.g. XNNPACK or gemmlowp a kernel with different rounding behavior
(with either single rounding or double rounding) is used this is impossible to
statically looking at the .tflite file and even the tensorflow MLIR based
compiler guesses what to do. To add even more insult to the injury
[FULLY_CONNECTED reference kernel should use MultiplyByQuantizedMultiplier
instead of std::round for requantization](https://github.com/tensorflow/tensorflow/issues/119412)
which means that this single operation has a semantic similar to the ONNX one
compared to the others which use the multiplier decomposition...

NOTE: all operator version in MLPerf Tiny 1.3 are <= 4
"""
import os
import pprint
from typing import Tuple, Set

import tflite

def is_tensor_constant(model: tflite.Model, tensor: tflite.Tensor) -> bool:
    buffer_index = tensor.Buffer()
    buffer = model.Buffers(buffer_index)
    return buffer is not None and buffer.DataLength() > 0

def analyze(model: tflite.Model) -> Tuple[int, int, int, Set[str], bool]:
    schema_version = model.Version()
    for i in range(tflite_model.MetadataLength()):
        meta = model.Metadata(i)
        if meta.Name().decode("utf-8") == "min_runtime_version":
            buffer_index = meta.Buffer()
            metadata = model.Buffers(buffer_index)
            min_runtime_version = metadata.DataAsNumpy().tobytes().decode('utf-8').rstrip('\x00')
            break
    print(schema_version, min_runtime_version)

    subgraph = model.Subgraphs(0)

    type_dict = {v: k for k, v in tflite.TensorType.__dict__.items() if not k.startswith('__')}

    float_types = {'FLOAT32', 'FLOAT16', 'FLOAT64'}
    int_types = {'INT8', 'UINT8', 'INT16', 'INT32', 'INT64'}

    input_indices = [subgraph.Inputs(i) for i in range(subgraph.InputsLength())]
    output_indices = [subgraph.Outputs(i) for i in range(subgraph.OutputsLength())]

    float_tensor_count = 0
    internal_float_tensor_count = 0
    int_quant_tensor_count = 0
    dtypes_found = set()
    uses_per_axis_quantization = False

    for i in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(i)
        tensor_type_name = type_dict[tensor.Type()]

        dtypes_found.add(tensor_type_name)

        if tensor_type_name in float_types:
            float_tensor_count += 1
            if i not in input_indices and i not in output_indices:
                internal_float_tensor_count += 1
        elif tensor_type_name in int_types:
            quantization = tensor.Quantization()
            # Otherwise the tensor could be something like the shape argument of
            # reshape.
            if quantization.ScaleLength() > 0:
                if quantization.ScaleLength() != 1:
                    uses_per_axis_quantization = True
                if not tensor_type_name.startswith("INT"):
                    raise ValueError("A UINT is needed...")
                int_quant_tensor_count += 1

    assert float_tensor_count >= internal_float_tensor_count
    is_integer_only = float_tensor_count == 0

    mixed_ops_count = 0
    weight_only_mixed_ops_count = 0
    ops_found = set()

    opcode_dict = {v: k for k, v in tflite.BuiltinOperator.__dict__.items() if not k.startswith('__')}
    operator_codes = [model.OperatorCodes(i).BuiltinCode() for i in range(model.OperatorCodesLength())]
    activation_dict = {
        v: k for k, v in tflite.ActivationFunctionType.__dict__.items()
        if not k.startswith('__')
    }
    builtin_options_dict = {
        v: k for k, v in tflite.BuiltinOptions.__dict__.items()
        if not k.startswith('__')
    }

    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)

        opcode_index = op.OpcodeIndex()
        op_code_enum = operator_codes[opcode_index]
        op_name = opcode_dict[op_code_enum]
        op_version = model.OperatorCodes(opcode_index).Version()

        print(op_name, op_version)

        ops_found.add(op_name)

        opt = op.BuiltinOptions()
        opt_type = op.BuiltinOptionsType()

        if opt is not None and opt_type != tflite.BuiltinOptions.NONE:
            class_name = builtin_options_dict.get(opt_type)
            OptionClass = getattr(tflite, class_name, None) if class_name else None

            if OptionClass is not None:
                options = OptionClass()
                options.Init(opt.Bytes, opt.Pos)

                if hasattr(options, 'FusedActivationFunction'):
                    fused_activation_enum = options.FusedActivationFunction()

                    if fused_activation_enum != tflite.ActivationFunctionType.NONE:
                        fused_op_name = f"FUSED_{activation_dict[fused_activation_enum]}"
                        ops_found.add(fused_op_name)

                if hasattr(options, 'QuantizedBiasType'):
                    if options.QuantizedBiasType() == tflite.TensorType.FLOAT32:
                        print("QuantizedBiasType: TensorType.FLOAT32")
                    else:
                        print("QuantizedBiasType: Other")

                if hasattr(options, 'PotScaleInt16'):
                    if options.PotScaleInt16():
                        print("PotScaleInt16: True")
                    else:
                        print("PotScaleInt16: False")


        if is_integer_only:
            continue

        has_float = False
        has_int_quant = False
        has_quantized_activation = False
        input_types_found = []

        for j in range(op.InputsLength()):
            tensor_idx = op.Inputs(j)

            is_optional = tensor_idx == -1
            if is_optional:
                continue

            tensor = subgraph.Tensors(tensor_idx)
            tensor_type_name = type_dict[tensor.Type()]
            input_types_found.append(tensor_type_name)

            if tensor_type_name in float_types:
                has_float = True
            elif tensor_type_name in int_types:
                quantization = tensor.Quantization()
                if quantization.ScaleLength() > 0:
                    has_int_quant = True

                    if not is_tensor_constant(model, tensor):
                        has_quantized_activation = True

        if has_float and has_int_quant:
            mixed_ops_count += 1

            if not has_quantized_activation:
                weight_only_mixed_ops_count += 1

    if mixed_ops_count != weight_only_mixed_ops_count: raise ValueError

    return float_tensor_count, int_quant_tensor_count, mixed_ops_count, ops_found, uses_per_axis_quantization

float_networks = []
int_quant_networks = []
mixed_networks = []
all_ops = set()
errors = []

for root, dirs, files in os.walk("3rdparty/tiny"):
    for file in files:
        if file.endswith(".tflite"):
            model_path = os.path.join(root, file)
            with open(model_path, "rb") as f:
                tflite_model_buf = f.read()
            tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
            (
                float_tensor_count, quant_tensors_count, mixed_ops_count,
                ops, per_axis_quant
            ) = analyze(tflite_model)
            print(file,
                "float_tensor_count = %d quant_tensors_count = %d mixed_ops_count = %d per_axis_quant %s"
                % (float_tensor_count, quant_tensors_count, mixed_ops_count, per_axis_quant))
            print(ops)
            all_ops.update(ops)
            if quant_tensors_count == 0:
                float_networks.append(model_path)
            elif mixed_ops_count == 0:
                int_quant_networks.append(model_path)
            else:
                mixed_networks.append(model_path)

print("all operations\n\t" + str(all_ops))
print("float_networks")
for path in float_networks: print("\t" + os.path.normpath(path))
print("int_quant_networks")
for path in int_quant_networks: print("\t" + os.path.normpath(path))
print("mixed_networks")
for path in mixed_networks: print("\t" + os.path.normpath(path))
print("errors")
for path in errors: print("\t" + os.path.normpath(path))
