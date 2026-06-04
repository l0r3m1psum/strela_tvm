import os
import pprint
from typing import Tuple

import tflite
import tvm.relax.frontend.tflite

def is_tensor_constant(model: tflite.Model, tensor: tflite.Tensor) -> bool:
    buffer_index = tensor.Buffer()
    buffer = model.Buffers(buffer_index)
    return buffer is not None and buffer.DataLength() > 0

def analyze(model: tflite.Model) -> Tuple[int, int, int]:
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
                if not tensor_type_name.startswith("INT"):
                    raise ValueError("A UINT is needed...")
                int_quant_tensor_count += 1

    assert float_tensor_count >= internal_float_tensor_count
    is_integer_only = float_tensor_count == 0

    mixed_ops_count = 0
    weight_only_mixed_ops_count = 0

    if not is_integer_only:
        opcode_dict = {v: k for k, v in tflite.BuiltinOperator.__dict__.items() if not k.startswith('__')}
        operator_codes = [model.OperatorCodes(i).BuiltinCode() for i in range(model.OperatorCodesLength())]

        for i in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(i)

            opcode_index = op.OpcodeIndex()
            op_code_enum = operator_codes[opcode_index]
            op_name = opcode_dict[op_code_enum]

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

    return float_tensor_count, int_quant_tensor_count, mixed_ops_count

float_networks = []
int_quant_networks = []
mixed_networks = []

for root, dirs, files in os.walk("3rdparty/tiny"):
    for file in files:
        if file.endswith(".tflite"):
            model_path = os.path.join(root, file)
            with open(model_path, "rb") as f:
                tflite_model_buf = f.read()
            tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
            float_tensor_count, quant_tensors_count, mixed_ops_count = analyze(tflite_model)
            print(file,
                "float_tensor_count = %d quant_tensors_count = %d mixed_ops_count = %d"
                % (float_tensor_count, quant_tensors_count, mixed_ops_count))
            if quant_tensors_count == 0:
                mod = tvm.relax.frontend.tflite.from_tflite(tflite_model)
                # mod.show()
                float_networks.append(model_path)
            elif mixed_ops_count == 0:
                int_quant_networks.append(model_path)
            else:
                mixed_networks.append(model_path)

print("float_networks")
for path in float_networks: print("\t" + os.path.normpath(path))
print("int_quant_networks")
for path in int_quant_networks: print("\t" + os.path.normpath(path))
print("mixed_networks")
for path in mixed_networks: print("\t" + os.path.normpath(path))
