import cv2
import time
import numpy as np
import onnxruntime as ort
import sys

def run_full_benchmark(model_path, image_path, iterations=8400):
    print(f"[INFO] Loading ONNX model in Python: {model_path}")
    
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Preprocessing (Not timed, exactly like C++)
    raw_frame = cv2.imread(image_path)
    resized = cv2.resize(raw_frame, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    input_tensor = np.transpose(normalized, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)

    print(f"[INFO] Warming up and running {iterations} iterations with Full Post-Processing...")
    
    latencies = []
    conf_threshold = 0.4
    nms_threshold = 0.45

    for i in range(iterations):
        start_time = time.perf_counter()
        
        # ==========================================
        # STAGE 3: INFERENCE
        # ==========================================
        outputs = session.run(None, {input_name: input_tensor})
        raw_output = outputs[0][0] # Shape: (84, 8400)
        
        # ==========================================
        # STAGE 4: POST-PROCESSING (YOLOv8/11 Logic)
        # ==========================================
        # Transpose to (8400, 84) to make it easier to parse
        raw_output = np.transpose(raw_output) 
        
        boxes = []
        confidences = []
        class_ids = []
        
        # We use numpy vectorization here to be as fair to Python as possible.
        # A pure Python 'for' loop over 8,400 elements would take over 100ms!
        class_scores = raw_output[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        best_class_ids = np.argmax(class_scores, axis=1)
        
        # Filter by confidence threshold
        mask = max_scores >= conf_threshold
        valid_predictions = raw_output[mask]
        valid_scores = max_scores[mask]
        valid_class_ids = best_class_ids[mask]
        
        # Convert bounding boxes
        for j in range(len(valid_predictions)):
            cx, cy, w, h = valid_predictions[j, 0:4]
            left = int(cx - w / 2)
            top = int(cy - h / 2)
            width = int(w)
            height = int(h)
            
            boxes.append([left, top, width, height])
            confidences.append(float(valid_scores[j]))
            class_ids.append(int(valid_class_ids[j]))
            
        # Non-Maximum Suppression (NMS)
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
            
        end_time = time.perf_counter()
        
        if i > 5:
            latencies.append((end_time - start_time) * 1000)

    avg_latency = sum(latencies) / len(latencies)
    print("======================================")
    print(f"FULL PYTHON PIPELINE BENCHMARK")
    print(f"Average Latency : {avg_latency:.2f} ms")
    print(f"Estimated FPS   : {1000.0 / avg_latency:.1f} FPS")
    print("======================================")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python benchmark_full.py <model.onnx> <image.jpg>")
    else:
        run_full_benchmark(sys.argv[1], sys.argv[2])