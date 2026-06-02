import os
import pprint

import tflite
import tvm.relax.frontend.tflite

imported = []
discarded = []

def analyze(model: tflite.Model):
    subgraph = model.Subgraphs(0)

    type_dict = {v: k for k, v in tflite.TensorType.__dict__.items() if not k.startswith('__')}

    float_types = {'FLOAT32', 'FLOAT16', 'FLOAT64'}
    int_types = {'INT8', 'UINT8', 'INT16', 'INT32', 'INT64'}

    input_indices = [subgraph.Inputs(i) for i in range(subgraph.InputsLength())]
    output_indices = [subgraph.Outputs(i) for i in range(subgraph.OutputsLength())]

    float_count = 0
    internal_float_tensors = 0
    int_count = 0
    dtypes_found = set()

    for i in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(i)
        tensor_type_name = type_dict[tensor.Type()]

        dtypes_found.add(tensor_type_name)

        if tensor_type_name in float_types:
            float_count += 1
            if i not in input_indices and i not in output_indices:
                internal_float_tensors += 1
        elif tensor_type_name in int_types:
            int_count += 1

    opcode_dict = {v: k for k, v in tflite.BuiltinOperator.__dict__.items() if not k.startswith('__')}
    operator_codes = [model.OperatorCodes(i).BuiltinCode() for i in range(model.OperatorCodesLength())]
    mixed_ops_found = 0

    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)

        opcode_index = op.OpcodeIndex()
        op_code_enum = operator_codes[opcode_index]
        op_name = opcode_dict[op_code_enum]

        has_float = False
        has_int = False
        input_types_found = []

        for j in range(op.InputsLength()):
            tensor_idx = op.Inputs(j)

            is_optional = tensor_idx == -1
            if is_optional:
                continue

            # Fetch the tensor and its type
            tensor = subgraph.Tensors(tensor_idx)
            tensor_type_name = type_dict.get(tensor.Type(), "UNKNOWN")
            input_types_found.append(tensor_type_name)

            if tensor_type_name in float_types:
                has_float = True
            elif tensor_type_name in int_types:
                has_int = True
                quantization = tensor.Quantization()
                assert quantization is not None
                scales = [quantization.Scale(i) for i in range(quantization.ScaleLength())]
                zero_points = [quantization.ZeroPoint(i) for i in range(quantization.ZeroPointLength())]

        if has_float and has_int:
            mixed_ops_found += 1
            print(f"⚠️ MIXED INPUTS in Operator {i} ({op_name})")
            print(f"   Input Types: {input_types_found}")

for root, dirs, files in os.walk("3rdparty/tiny"):
    for file in files:
        if file.endswith(".tflite"):
            model_path = os.path.join(root, file)
            with open(model_path, "rb") as f:
                tflite_model_buf = f.read()
            tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
            analyze(tflite_model)
            if (
                "_quant" not in file
                and "_int8" not in file
                and "kws_ref_model.tflite" != file
                and "kws_ref_model_float32.tflite" != file
                and "str_ww_ref_model.tflite" != file
            ):
                print(model_path)

                mod = tvm.relax.frontend.tflite.from_tflite(tflite_model)
                mod.show()
                imported.append(model_path)
            else:
                discarded.append(model_path)

print("imported")
pprint.pprint(imported)
print("discarded")
pprint.pprint(discarded)
