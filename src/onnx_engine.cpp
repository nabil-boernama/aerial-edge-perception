#include "inference_engine.hpp"

#include <iostream>
#include <utility>

OnnxRuntimeEngine::OnnxRuntimeEngine() :
    env_(ORT_LOGGING_LEVEL_WARNING, "EdgeEngine"),
    memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    session_options_.SetIntraOpNumThreads(4);
    session_options_.SetInterOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

bool OnnxRuntimeEngine::loadModel(const std::string& model_path) 
{
    try
    {
        // 1. Initialize the session
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

        // 2. Get the number of inputs and outputs
        size_t num_input_nodes = session_->GetInputCount();
        size_t num_output_nodes = session_->GetOutputCount();

        // The allocator is used to extract strings from the ONNX model memory safely
        Ort::AllocatorWithDefaultOptions allocator;

        // ==========================================
        // 3. EXTRACT INPUT METADATA (Names & Shapes)
        // ==========================================
        for (size_t i = 0; i < num_input_nodes; i++) 
        {
            // Extract the name of the input node
            Ort::AllocatedStringPtr input_name_ptr = session_->GetInputNameAllocated(i, allocator);
            
            // Copy it into our C++ std::string vector so it doesn't get destroyed
            input_node_names_.push_back(input_name_ptr.get());
            
            // Store a pointer to the safe string for later use during inference
            input_node_names_ptr_.push_back(input_node_names_.back().c_str());

            // Extract the shape of the input tensor (e.g., 1 x 3 x 640 x 640)
            Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = tensor_info.GetShape();

            // Edge AI Trick: YOLO models often have dynamic batch sizes (a -1 at index 0).
            // We force it to 1 because our drone only processes one frame at a time.
            if (shape[0] == -1) {
                shape[0] = 1;
            }
            input_shape_ = shape; // Save to our class variable
        }

        // ==========================================
        // 4. EXTRACT OUTPUT METADATA (Names & Shapes)
        // ==========================================
        for (size_t i = 0; i < num_output_nodes; i++) 
        {
            // Extract the name of the output node
            Ort::AllocatedStringPtr output_name_ptr = session_->GetOutputNameAllocated(i, allocator);
            
            output_node_names_.push_back(output_name_ptr.get());
            output_node_names_ptr_.push_back(output_node_names_.back().c_str());

            // Extract the shape of the output tensor (e.g., 1 x 84 x 8400)
            Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = tensor_info.GetShape();

            // Handle dynamic batch size for the output as well
            if (shape[0] == -1) {
                shape[0] = 1;
            }
            output_shape_ = shape; 
        }

        // If we made it here without throwing an exception, the model loaded perfectly!
        std::cout << "Successfully loaded ONNX model: " << model_path << std::endl;
        return true;

    }
    catch (const Ort::Exception& e) 
    {
        // Never swallow an exception! Print exactly why it failed.
        std::cerr << "ONNX Runtime Exception during model load: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e)
    {
        // Catch standard C++ errors (like out of memory)
        std::cerr << "Standard C++ Exception: " << e.what() << std::endl;
        return false;
    }
}

std::vector<Detection> OnnxRuntimeEngine::runInference(const std::vector<float>& input_tensor) const
{
    // 1. Create ORT Tensor from the input vector (Zero-copy wrapper)
    // Note: The ORT API expects a non-const pointer, even for inputs.
    // We use const_cast safely here because session_->Run() will not modify our input data.
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_ort_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input_tensor.data()), // Pointer to the raw image numbers
        input_tensor.size(),                     // Total number of floats (e.g., 3x640x640)
        input_shape_.data(),                     // Shape array (e.g., {1, 3, 640, 640})
        input_shape_.size()
    );

    // 2. Run the actual forward pass!
    // This is the bottleneck line. When this runs, the Pi 5 NEON cores spike to 100%.
    std::vector<Ort::Value> output_tensors = session_->Run(
        Ort::RunOptions{nullptr},          // Default run options
        input_node_names_ptr_.data(),      // Names of the input nodes
        &input_ort_tensor,                 // Pointer to our zero-copy input tensor
        1,                                 // Number of inputs (1 image)
        output_node_names_ptr_.data(),     // Names of the output nodes
        output_node_names_ptr_.size()      // Number of outputs
    );

    // 3. Extract the raw output data
    float* raw_output = output_tensors[0].GetTensorMutableData<float>();
    
    // Grab the shape dynamically so we know how to parse the matrix
    auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> out_shape = output_info.GetShape(); 

    std::vector<Detection> detections;
    
    // ==========================================
    // STAGE 4 - POST-PROCESSING (NMS) for YOLOv8 / YOLO11
    // ==========================================
    
    // We are looking for output shape: [1, 84, 8400]
    // out_shape[1] = 84 (4 box coords + 80 classes)
    // out_shape[2] = 8400 (number of predictions)
    int num_classes = out_shape[1] - 4; // usually 80 for COCO
    int num_predictions = out_shape[2]; // usually 8400

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    // Minimum confidence threshold (e.g., 0.4 means 40% sure)
    const float CONFIDENCE_THRESHOLD = 0.4f;

    // Loop through all 8400 predictions
    for (int i = 0; i < num_predictions; ++i) 
    {
        float max_class_score = 0.0f;
        int best_class_id = -1;

        // Since the matrix is transposed (84 x 8400), we jump across the array 
        // to check all 80 class scores for this specific prediction 'i'
        for (int c = 0; c < num_classes; ++c) 
        {
            // The 1D array index math: (row * total_columns) + current_column
            // Class scores start at row 4.
            float score = raw_output[(4 + c) * num_predictions + i];
            
            if (score > max_class_score) 
            {
                max_class_score = score;
                best_class_id = c;
            }
        }

        // If the best class score beats our threshold, save the box!
        if (max_class_score >= CONFIDENCE_THRESHOLD) 
        {
            // Extract the box coordinates (Rows 0, 1, 2, 3 for prediction 'i')
            float cx = raw_output[0 * num_predictions + i]; // Center X
            float cy = raw_output[1 * num_predictions + i]; // Center Y
            float w  = raw_output[2 * num_predictions + i]; // Width
            float h  = raw_output[3 * num_predictions + i]; // Height

            // Convert Center X, Y to Top-Left X, Y for OpenCV Rect
            int left = static_cast<int>(cx - (w / 2.0f));
            int top = static_cast<int>(cy - (h / 2.0f));
            int width = static_cast<int>(w);
            int height = static_cast<int>(h);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(max_class_score);
            class_ids.push_back(best_class_id);
        }
    }

    // --- Non-Maximum Suppression (NMS) ---
    // The model will predict 10 boxes for the same target. NMS merges them.
    std::vector<int> nms_indices;
    const float NMS_THRESHOLD = 0.45f; // How much overlap is allowed before deleting a box
    cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, nms_indices);

    // Build the final clean list of detections to return
    std::vector<Detection> detections;
    for (int idx : nms_indices) 
    {
        Detection det;
        det.box = boxes[idx];
        det.confidence = confidences[idx];
        det.class_id = class_ids[idx];
        detections.push_back(det);
    }

    return detections;
}