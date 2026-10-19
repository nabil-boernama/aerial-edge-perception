// Copyright 2026 Your Name
// SPDX-License-Identifier: MIT

// Interface classes for deep learning inference on edge SBCs.

#ifndef AERIAL_EDGE_PERCEPTION_INFERENCE_ENGINE_H_
#define AERIAL_EDGE_PERCEPTION_INFERENCE_ENGINE_H_

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// Plain-old-data result for a single detected object.
struct Detection {
    cv::Rect box;      // Bounding box in pixels (x, y, w, h).
    float confidence;  // [0.0, 1.0].
    int class_id;      // Numeric class index.
};

// Abstract interface for a deep-learning inference backend.
//
// Derived classes must be thread-safe with respect to concurrent
// calls to runInference().
class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;

    // Loads a model from disk into the inference session.
    //
    // WARNING: Performs disk I/O and heap allocation. Do not call
    // from a real-time control loop.
    //
    // Returns true if the model was loaded successfully.
    [[nodiscard]] virtual bool loadModel(const std::string& model_path) = 0;

    // Runs a single forward pass on a preprocessed image.
    //
    // input_tensor must be RGB, NCHW (1x3x640x640), normalized to [0, 1].
    //
    // Returns detections that passed Non-Maximum Suppression (NMS) and the confidence threshold.
    [[nodiscard]] virtual std::vector<Detection> runInference(
        const std::vector<float>& input_tensor) const = 0;
};

// ONNX Runtime C++ backend, optimized for CPU (ARM NEON on RPi 5).
class OnnxRuntimeEngine : public InferenceEngine {
public:
    OnnxRuntimeEngine();
    ~OnnxRuntimeEngine() override = default;

    [[nodiscard]] bool loadModel(const std::string& model_path) override;
    [[nodiscard]] std::vector<Detection> runInference(
        const std::vector<float>& input_tensor) const override;

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;

    // Owned strings; pointers below reference their c_str() data.
    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;
    std::vector<const char*> input_node_names_ptr_;
    std::vector<const char*> output_node_names_ptr_;

    // Cached tensor shapes to avoid re-querying the session per frame.
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
};

#endif  // AERIAL_EDGE_PERCEPTION_INFERENCE_ENGINE_H_   