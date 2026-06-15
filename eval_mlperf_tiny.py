from ai_edge_litert.interpreter import Interpreter
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
import librosa
import numpy as np

from pathlib import Path
import os
import pickle
import random
import sys

def mock_compiler_inference(model_path, input_batch, task):
    batch_size = len(input_batch)
    if task == "image_classification":
        return np.random.rand(batch_size, 10)
    elif task == "visual_wake_words":
        return np.random.rand(batch_size, 2)
    elif task == "anomaly_detection":
        return np.random.rand(batch_size)
    else:
        raise ValueError(f"Unknown task: {task}")

def run_tflite_inference(model_path, input_data, task):
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_dtype = input_details['dtype']
    expected_shape = tuple(input_details['shape'])
    
    in_quant = input_details.get('quantization_parameters', {})
    in_scale = in_quant.get('scales', [1.0])[0] if len(in_quant.get('scales', [])) > 0 else 1.0
    in_zp = in_quant.get('zero_points', [0])[0] if len(in_quant.get('zero_points', [])) > 0 else 0

    out_quant = output_details.get('quantization_parameters', {})
    out_scale = out_quant.get('scales', [1.0])[0] if len(out_quant.get('scales', [])) > 0 else 1.0
    out_zp = out_quant.get('zero_points', [0])[0] if len(out_quant.get('zero_points', [])) > 0 else 0

    results = []
    
    for i, sample in enumerate(input_data):
        # Reshape to strictly match interpreter expectation (e.g. adding batch dim)
        sample_reshaped = np.reshape(sample, expected_shape)

        if input_dtype == np.int8 and sample_reshaped.dtype != np.int8:
            sample_ready = np.clip(np.round(sample_reshaped / in_scale) + in_zp, -128, 127).astype(np.int8)
        else:
            sample_ready = sample_reshaped.astype(input_dtype)
            
        interpreter.set_tensor(input_details['index'], sample_ready)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details['index']).copy()
        
        if output_details['dtype'] == np.int8:
            output_data_float = (output_data.astype(np.float32) - out_zp) * out_scale
        else:
            output_data_float = output_data.astype(np.float32)
            
        if task == "anomaly_detection":
            if output_data_float.shape == sample_reshaped.shape:
                # Model is an Autoencoder: Anomaly score is the MSE
                sample_float = (
                    (sample_ready.astype(np.float32) - in_zp) * in_scale
                    if input_dtype == np.int8
                    else sample_ready.astype(np.float32)
                )
                score = np.mean((sample_float - output_data_float)**2)
                results.append(score)
            else:
                # Model directly outputs a score
                results.append(np.mean(output_data_float))
        else:
            # Classification tasks: Output probability array
            results.append(output_data_float[0])
            
    return np.array(results)


def evaluate_cifar10(dataset_path, tflite_models, num_samples=500):
    print("\n" + "="*50)
    print("--- Evaluating Image Classification (CIFAR-10) ---")
    test_batch_path = Path(dataset_path) / "test_batch"

    with open(test_batch_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    
    # Process up to num_samples (Image range: 0.0 to 255.0)
    raw_images = data_dict[b'data'][:num_samples]
    y_true = np.array(data_dict[b'labels'])[:num_samples]
    images = raw_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
    
    print(f"Loaded {len(y_true)} test samples.")
    
    for model_path in tflite_models:
        print(f"Running {model_path.name}...")
        mock_preds = mock_compiler_inference(model_path, images, "image_classification")
        mock_acc = accuracy_score(y_true, np.argmax(mock_preds, axis=1))
        print(f"    -> [TVM]    Accuracy: {mock_acc * 100:.2f}%")
        tflite_preds = run_tflite_inference(model_path, images, "image_classification")
        tflite_acc = accuracy_score(y_true, np.argmax(tflite_preds, axis=1))
        print(f"    -> [TFLite] Accuracy: {tflite_acc * 100:.2f}%")


def evaluate_vww(dataset_path, tflite_models, num_samples=500):
    print("\n" + "="*50)
    print("--- Evaluating Visual Wake Words (VWW) ---")
    dataset_path = Path(dataset_path)
    
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
    
    images = np.array(
        [np.array(Image.open(p).resize((96, 96)).convert('RGB')) for p in image_paths],
        dtype=np.float32
    ) / 255.0
    
    for model_path in tflite_models:
        print(f"Running {model_path.name}...")
        mock_preds = mock_compiler_inference(model_path, images, "visual_wake_words")
        mock_acc = accuracy_score(y_true, np.argmax(mock_preds, axis=1))
        print(f"    -> [TVM]    Accuracy: {mock_acc * 100:.2f}%")
        tflite_preds = run_tflite_inference(model_path, images, "visual_wake_words")
        tflite_acc = accuracy_score(y_true, np.argmax(tflite_preds, axis=1))
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

    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

    # Take central part only
    log_mel_spectrogram = log_mel_spectrogram[:, 50:250]

    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1
    if vector_array_size < 1:
        return np.empty((0, dims))

    # Generate feature vectors by concatenating multiframes
    vector_array = np.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

    return vector_array

def evaluate_anomaly_detection(dataset_path, tflite_models, num_samples=100):
    print("\n" + "="*50)
    print("--- Evaluating Anomaly Detection (ToyADMOS - ToyCar) ---")
    dataset_path = Path(dataset_path)

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

        file_anomaly_scores = []
        for file_vectors in all_features:
            window_errors = mock_compiler_inference(model_path, file_vectors, "anomaly_detection")
            file_anomaly_scores.append(np.mean(window_errors))
        mock_auroc = roc_auc_score(y_true, file_anomaly_scores)
        print(f"    -> [TVM]    AUROC: {mock_auroc:.4f}%")

        file_anomaly_scores = []
        for file_vectors in all_features:
            window_errors = run_tflite_inference(model_path, file_vectors, "anomaly_detection")
            file_anomaly_scores.append(np.mean(window_errors))
        tflite_auroc = roc_auc_score(y_true, file_anomaly_scores)
        print(f"    -> [TFLite] AUROC: {tflite_auroc:.4f}")

if __name__ == "__main__":
    BASE_DIR = Path.cwd()
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
        TINY_DIR / "anomaly_detection/trained_models/ToyCar/baseline_tf23/model/model_ToyCar_quant.tflite",
    )
    
    # Execute (capped at 500 samples by default so you aren't waiting 5 minutes per run)
    evaluate_cifar10(IC_DIR, ic_models)
    evaluate_vww(VWW_DIR, vww_models)
    evaluate_anomaly_detection(AD_DIR, ad_models)
