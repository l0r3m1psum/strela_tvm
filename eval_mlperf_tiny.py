from ai_edge_litert.interpreter import Interpreter
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
import librosa
import numpy
import tflite
import tvm.relax.frontend.tflite
from tvm import relax

import pathlib
import os
import pickle
import random
import sys

from typing import Dict, List, Tuple

def get_io_quantization_params(
    func: relax.Function
) -> Tuple[Dict[relax.Var, Tuple], List[Tuple]]:
    """
    Scans a Relax function and returns the parameters of the first (if any)
    dequantize operation applied to each function argument and the last (if any)
    quantize operation applied to each function output.

    This is needed to recover the quantization parameters to correctly input
    data and successively interpret the outputs of the network when both input
    and output are quantized.

    The function returns the parameters of the first dequantize after any number
    of quantization invariant operations and the last quantize before any number
    of quantization invariant operations.

    An input could be dequantized in two ways at once i.e. let i be an input
    and dq1 and dq2 be two dequantize function with different parameters if we
    have a dataflow graph like this

        i
       / \
    dq1   dq2

    We have to throw an error since there is no way to quantize the same input
    in two different ways.

    Quantization should really be done at the type level or at least be attached
    to the tensors because in this way it is a mess...

    Returns:
        input_params: dict mapping relax.Var (input param) to its (scale, zero_point, axis, dtype)
        output_params: list of (scale, zero_point, axis, dtype) for each output tuple element
    """
    try:
        df_block = func.body.blocks[0]
        # Given a variable v, udchain[v] is the list of variable that it is used
        # to define e.g.
        # lv2 = f(lv, lv1)
        # lv3 = g(lv1)     => {lv: (lv2, lv4), lv1: (lv2, lv3)}
        # lv4 = g(lv)
        udchain: Dict[relax.Var, Tuple[relax.Var]] = relax.analysis.udchain(df_block)
    except (TypeError, IndexError, AttributeError) as ex:
        raise ValueError(
            "The function should be in dataflow form i.e. a SeqExpr with "
            "a DataflowBlock as its first block."
        ) from ex

    # Given a variable v, duchain[v] is the expression that defined it
    # lv1 = f(lv) => {lv1: f(lv)}
    # Because of SSA, we just map every bound variable to its value expression.
    duchain: Dict[relax.Var, relax.Expr] = {binding.var: binding.value for binding in df_block.bindings}

    # Operations that only manipulate shapes, not the quantized values
    invariant_ops = {
        "relax.reshape", "relax.permute_dims", "relax.expand_dims",
        "relax.squeeze", "relax.strided_slice", "relax.split",

        # "relax.nn.relu", "relax.clip", "relax.maximum", "relax.minimum",
    }

    def extract_constant_numpy(expr):
        return expr.data.numpy() if isinstance(expr, relax.Constant) else expr

    def params_are_equal(p1, p2):
        """Compares two sets of quantization parameters."""
        s1, z1, a1, d1 = p1
        s2, z2, a2, d2 = p2
        if a1 != a2 or d1 != d2:
            return False

        def expr_eq(e1, e2):
            return numpy.array_equal(e1, e2) \
                if isinstance(e1, numpy.ndarray) and isinstance(e2, numpy.ndarray) \
                else e1 == e2

        return expr_eq(s1, s2) and expr_eq(z1, z2)

    input_qparams = {}

    def trace_forward(var):
        """Traces through all the uses of the uses of var tracing deeper only if
        there is an quantization invariant operations or TupleGetItem. In this
        example i is used by three operations of which one is traced throw
        because it is quantization invariant.
             i
           / | \
        dq1 inv f
             |
            dq2
        """
        found = []
        for use_var in udchain.get(var, []):
            expr = duchain.get(use_var, None)
            if isinstance(expr, relax.Call):
                op_name = expr.op.name if hasattr(expr.op, "name") else ""
                if op_name == "relax.dequantize":
                    scale = extract_constant_numpy(expr.args[1])
                    zp = extract_constant_numpy(expr.args[2])
                    axis = expr.attrs.axis if hasattr(expr.attrs, "axis") else -1
                    out_dt = expr.attrs.out_dtype if hasattr(expr.attrs, "out_dtype") else "float32"
                    found.append((scale, zp, axis, out_dt))
                elif op_name in invariant_ops:
                    found.extend(trace_forward(use_var))
            elif isinstance(expr, relax.TupleGetItem):
                found.extend(trace_forward(use_var))
        return found

    for p in func.params:
        found_params = trace_forward(p)

        if found_params:
            # Conflict Resolution logic
            unique_params = []
            for qp in found_params:
                if not any(params_are_equal(qp, up) for up in unique_params):
                    unique_params.append(qp)

            if len(unique_params) > 1:
                raise ValueError(
                    f"Conflict detected: Input '{p.name_hint}' is dequantized with "
                    f"multiple conflicting parameters: {unique_params}"
                )

            input_qparams[p] = unique_params[0]
        else:
            input_qparams[p] = ()

    def trace_backward(var):
        expr = duchain.get(var, None)

        if isinstance(expr, relax.Call):
            op_name = expr.op.name if hasattr(expr.op, "name") else ""
            if op_name == "relax.quantize":
                scale = extract_constant_numpy(expr.args[1])
                zp = extract_constant_numpy(expr.args[2])
                axis = expr.attrs.axis if hasattr(expr.attrs, "axis") else -1
                out_dt = expr.attrs.out_dtype if hasattr(expr.attrs, "out_dtype") else "int8"
                return [(scale, zp, axis, out_dt)]
            elif op_name in invariant_ops:
                if isinstance(expr.args[0], relax.Var):
                    return trace_backward(expr.args[0])
        elif isinstance(expr, relax.TupleGetItem):
            if isinstance(expr.tuple_value, relax.Var):
                return trace_backward(expr.tuple_value)

        return []

    out_expr = func.body.body
    out_vars = []
    if isinstance(out_expr, relax.Var):
        out_vars = [out_expr]
    elif isinstance(out_expr, relax.Tuple):
        out_vars = [f for f in out_expr.fields if isinstance(f, relax.Var)]

    output_qparams = []
    for out_v in out_vars:
        found_params = trace_backward(out_v)
        output_qparams.append(found_params[0] if found_params else ())

    return input_qparams, output_qparams

def quantize(x, s, zp):
    return numpy.clip(numpy.round(x/s) + zp, -128, 127).astype(numpy.int8)

def dequantize(q, s, zp):
    return (q.astype(numpy.float32) - zp)*s

def run_tflite_inference(model_path, input_batches, task):
    """
    For Classification: input_batches is a list of image batches.
    For Anomaly Detection: input_batches is a list where each element is
                           all sliding windows for a SINGLE audio file.
    """
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    with open(model_path, "rb") as f:
        tflite_model_buf = f.read()
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    mod = relax.frontend.tflite.from_tflite(tflite_model)
    # mod = relax.transform.NormalizeQDQPatterns()(mod)
    inputs_qparams, outputs_qparams = get_io_quantization_params(mod["main"])
    input_qparams = next(iter(inputs_qparams.values()))
    output_qparams = outputs_qparams[0]
    # mod = relax.transform.RewriteQDQPatternsToQNNOps()(mod)
    # mod = relax.transform.LowerQNNOps()(mod)
    ex = tvm.compile(mod, tvm.target.Target("llvm"))
    vm = relax.VirtualMachine(ex, tvm.cpu())

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_dtype = input_details['dtype']
    expected_shape = tuple(input_details['shape']) # e.g., (1, 640) or (1, 96, 96, 3)

    in_quant = input_details.get('quantization_parameters', {})
    in_scale_tf = in_quant.get('scales', [1.0])[0] if len(in_quant.get('scales', [])) > 0 else 1.0
    in_zp_tf = in_quant.get('zero_points', [0])[0] if len(in_quant.get('zero_points', [])) > 0 else 0

    out_quant = output_details.get('quantization_parameters', {})
    out_scale_tf = out_quant.get('scales', [1.0])[0] if len(out_quant.get('scales', [])) > 0 else 1.0
    out_zp_tf = out_quant.get('zero_points', [0])[0] if len(out_quant.get('zero_points', [])) > 0 else 0

    if input_qparams:
        in_scale_tvm, in_zp_tvm, in_axis_tvm, in_out_dtype_tvm = input_qparams
    else:
        in_scale_tvm, in_zp_tvm, in_axis_tvm, in_out_dtype_tvm = 1.0, 0, -1, "float32"

    if output_qparams:
        out_scale_tvm, out_zp_tvm, out_axis_tvm, out_out_dtype_tvm = output_qparams
    else:
        out_scale_tvm, out_zp_tvm, out_axis_tvm, out_out_dtype_tvm = 1.0, 0, -1, "int8"

    if in_scale_tf != in_scale_tvm or in_zp_tf != in_zp_tvm:
        raise RuntimeError(
            "Input quantization parameters differ from LiteRT to TVM %f %d vs %f %d"
            % (in_scale_tf, in_zp_tf, in_scale_tvm, in_zp_tvm)
        )

    if out_scale_tf != out_scale_tvm or out_zp_tf != out_zp_tvm:
        raise RuntimeError(
            "Output quantization parameters differ from LiteRT to TVM %f %d vs %f %d"
            % (out_scale_tf, out_zp_tf, out_scale_tvm, out_zp_tvm)
        )

    results_tf = []
    results_tvm = []
    current_batch_size = expected_shape[0]

    for batch in input_batches:
        B = len(batch)
        target_shape = [B] + list(expected_shape[1:])

        if target_shape[0] != current_batch_size:
            interpreter.resize_tensor_input(input_details['index'], target_shape)
            interpreter.allocate_tensors()

            # Refresh details after re-allocation as memory pointers may have shifted
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]
            current_batch_size = target_shape[0]

        batch_reshaped = numpy.reshape(batch, target_shape)

        if input_dtype == numpy.int8 and batch_reshaped.dtype != numpy.int8:
            batch_ready = quantize(batch_reshaped, in_scale_tf, in_zp_tf)
        else:
            batch_ready = batch_reshaped.astype(input_dtype)

        interpreter.set_tensor(input_details['index'], batch_ready)
        interpreter.invoke()
        output_data_tf = interpreter.get_tensor(output_details['index']).copy()

        output_data_tvm = []
        for datum in batch_ready:
            datum = datum[numpy.newaxis, :]
            datum = tvm.runtime.tensor(datum, tvm.cpu())
            out = vm["main"](datum)
            output_data_tvm.append(out.numpy())
        output_data_tvm = numpy.concatenate(output_data_tvm, axis=0)


        for output_data, results in ((output_data_tf, results_tf,), (output_data_tvm, results_tvm,),):
            if output_data.dtype == numpy.int8:
                output_data_float = dequantize(output_data, out_scale_tf, out_zp_tf)
            else:
                output_data_float = output_data.astype(numpy.float32)

            if task == "anomaly_detection":
                if output_data_float.shape == batch_reshaped.shape:
                    # Model is an Autoencoder: Anomaly score is the Mean Squared Error (MSE)
                    batch_float = (
                        dequantize(batch_ready, in_scale_tf, in_zp_tf)
                        if input_dtype == numpy.int8
                        else batch_ready.astype(numpy.float32)
                    )

                    # Compute the single file anomaly score (MSE averaged across ALL windows in the batch)
                    score = numpy.mean((batch_float - output_data_float)**2)
                    results.append(score)
                else:
                    # Model directly outputs a score (Mean of scores if multiple windows)
                    results.append(numpy.mean(output_data_float))
            else:
                # Classification tasks: Extend list to keep individual sample probabilities flat
                results.extend(output_data_float)
            
    return numpy.array(results_tf), numpy.array(results_tvm)

def evaluate_cifar10(dataset_path, tflite_models, num_samples=500):
    print("\n" + "="*50)
    print("--- Evaluating Image Classification (CIFAR-10) ---")
    test_batch_path = pathlib.Path(dataset_path) / "test_batch"

    with open(test_batch_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    
    # Process up to num_samples (Image range: 0.0 to 255.0)
    raw_images = data_dict[b'data'][:num_samples]
    y_true = numpy.array(data_dict[b'labels'])[:num_samples]
    images = raw_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(numpy.float32)
    
    print(f"Loaded {len(y_true)} test samples.")
    
    for model_path in tflite_models:
        print(f"Running {model_path.name}...")
        tflite_preds, mock_preds = run_tflite_inference(model_path, [images], "image_classification")
        mock_acc = accuracy_score(y_true, numpy.argmax(mock_preds, axis=1))
        print(f"    -> [TVM]    Accuracy: {mock_acc * 100:.2f}%")
        tflite_acc = accuracy_score(y_true, numpy.argmax(tflite_preds, axis=1))
        print(f"    -> [TFLite] Accuracy: {tflite_acc * 100:.2f}%")

def evaluate_vww(dataset_path, tflite_models, num_samples=500):
    print("\n" + "="*50)
    print("--- Evaluating Visual Wake Words (VWW) ---")
    dataset_path = pathlib.Path(dataset_path)
    
    # Extract equal amounts of 'person' and 'not_person'
    half_samples = num_samples // 2
    categories = {"not_person": 0, "person": 1}
    image_paths = []
    y_true = []
    
    for cat_name, label in categories.items():
        cat_dir = dataset_path / cat_name
        if cat_dir.exists():
            paths = list(cat_dir.glob("*.jpg"))[:half_samples]
            image_paths.extend(paths)
            y_true.extend([label] * len(paths))
    
    if not image_paths:
        print("Error: No images found.")
        return
        
    print(f"Loaded {len(image_paths)} test images.")
    
    images = numpy.array(
        [numpy.array(Image.open(p).resize((96, 96)).convert('RGB')) for p in image_paths],
        dtype=numpy.float32
    ) / 255.0
    
    for model_path in tflite_models:
        print(f"Running {model_path.name}...")
        tflite_preds, mock_preds = run_tflite_inference(model_path, [images], "visual_wake_words")
        mock_acc = accuracy_score(y_true, numpy.argmax(mock_preds, axis=1))
        print(f"    -> [TVM]    Accuracy: {mock_acc * 100:.2f}%")
        tflite_acc = accuracy_score(y_true, numpy.argmax(tflite_preds, axis=1))
        print(f"    -> [TFLite] Accuracy: {tflite_acc * 100:.2f}%")

def file_to_vector_array(file_name, n_mels=128, frames=5, n_fft=1024, hop_length=512, power=2.0):
    """
    Extracts sliding-window log-mel spectrogram features.
    This guarantees exact HTK/Slaney parity with the MLPerf trained models.
    """
    dims = n_mels * frames
    y, sr = librosa.load(file_name, sr=None)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power
    )

    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    # Take central part only
    log_mel_spectrogram = log_mel_spectrogram[:, 50:250]

    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1
    if vector_array_size < 1:
        return numpy.empty((0, dims))

    # Generate feature vectors by concatenating multiframes
    vector_array = numpy.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

    return vector_array

def evaluate_anomaly_detection(dataset_path, tflite_models, num_samples=100):
    print("\n" + "="*50)
    print("--- Evaluating Anomaly Detection (ToyADMOS - ToyCar) ---")
    dataset_path = pathlib.Path(dataset_path)

    wav_files = list(dataset_path.rglob("ToyCar/test/*.wav"))

    if not wav_files:
        print("Error: Could not find any .wav files in the ToyCar directory.")
        return

    normal_files = [f for f in wav_files if f.name.startswith("normal")]
    anomaly_files = [f for f in wav_files if f.name.startswith("anomaly")]

    # I don't know why shuffling improves result....
    random.seed(42)
    random.shuffle(normal_files)
    random.shuffle(anomaly_files)

    half = num_samples // 2
    selected_files = normal_files[:half] + anomaly_files[:half]
    raw_y_true = [0] * len(normal_files[:half]) + [1] * len(anomaly_files[:half])

    print(f"Discovered {len(raw_y_true)} test audio files. Extracting Librosa Log-Mel features...")

    all_features = []
    y_true = []

    for i, file_path in enumerate(selected_files):
        try:
            vectors = file_to_vector_array(file_path, n_mels=128, frames=5)
            if vectors.shape[0] > 0:
                all_features.append(vectors)
                y_true.append(raw_y_true[i])
        except Exception as e:
            print(f"  [Warning] Failed to process {file_path.name}: {e}")

    print(f"Successfully processed {len(all_features)} files.")

    for model_path in tflite_models:
        print(f"   Running {model_path.name}...")
        tflite_scores, mock_scores = run_tflite_inference(model_path, all_features, "anomaly_detection")
        mock_auroc = roc_auc_score(y_true, mock_scores)
        print(f"    -> [TVM]    AUROC: {mock_auroc:.4f}")
        tflite_auroc = roc_auc_score(y_true, tflite_scores)
        print(f"    -> [TFLite] AUROC: {tflite_auroc:.4f}")

if __name__ == "__main__":
    BASE_DIR = pathlib.Path.cwd()
    TINY_DIR = BASE_DIR / "3rdparty/tiny/benchmark/training"
    
    AD_DIR  = TINY_DIR / "anomaly_detection/dev_data"
    IC_DIR  = TINY_DIR / "image_classification/cifar-10-batches-py"
    VWW_DIR = TINY_DIR / "visual_wake_words/vw_coco2014_96"
    
    ic_models = (
        TINY_DIR / "image_classification/trained_models/pretrainedResnet_large_int8.tflite",
        TINY_DIR / "image_classification/trained_models/pretrainedResnet_quant.tflite",
    )
    
    vww_models = (
        TINY_DIR / "visual_wake_words/trained_models/vww_96_int8.tflite",
    )
    
    ad_models = (
        TINY_DIR / "anomaly_detection/trained_models/ad01_int8.tflite",
        TINY_DIR / "anomaly_detection/trained_models/ToyCar/baseline_tf23/model/model_ToyCar_quant_fullint.tflite",
        TINY_DIR / "anomaly_detection/trained_models/ToyCar/baseline_tf23/model/model_ToyCar_quant_fullint_micro.tflite",
        TINY_DIR / "anomaly_detection/trained_models/ToyCar/baseline_tf23/model/model_ToyCar_quant_fullint_micro_intio.tflite",
        # TINY_DIR / "anomaly_detection/trained_models/ToyCar/baseline_tf23/model/model_ToyCar_quant.tflite",
    )
    
    # Execute (capped at 500 samples by default so you aren't waiting 5 minutes per run)
    evaluate_cifar10(IC_DIR, ic_models)
    # TODO: support depthwise convolution in the tflite importer.
    # evaluate_vww(VWW_DIR, vww_models)
    evaluate_anomaly_detection(AD_DIR, ad_models)
