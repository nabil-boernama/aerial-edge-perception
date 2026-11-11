#include "inference_engine.hpp"

#include <iostream>
#include <vector>
#include <chrono> // For benchmarking
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) 
{
    if (argc < 3) {
        std::cerr << "Usage: ./aero_edge_cv <path_to_model.onnx> <path_to_image.jpg>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];

    // ==========================================
    // 1. INITIALIZE ENGINE
    // ==========================================
    std::cout << "[INFO] Initializing ONNX Runtime Engine..." << std::endl;
    OnnxRuntimeEngine engine;
    
    if (!engine.loadModel(model_path)) {
        std::cerr << "[ERROR] Failed to load model. Exiting." << std::endl;
        return -1;
    }

    // ==========================================
    // 2. SIMULATE CAMERA ACQUISITION
    // ==========================================
    cv::Mat raw_frame = cv::imread(image_path);
    if (raw_frame.empty()) {
        std::cerr << "[ERROR] Could not read image: " << image_path << std::endl;
        return -1;
    }
    
    std::cout << "[INFO] Image loaded. Original size: " 
              << raw_frame.cols << "x" << raw_frame.rows << std::endl;

    // ==========================================
    // 3. PREPROCESSING (HWC to NCHW)
    // ==========================================
    // YOLO models typically expect 640x640
    const int target_width = 640;
    const int target_height = 640;

    // blobFromImage handles resize, BGR->RGB, normalization (1.0/255.0), and HWC->NCHW
    cv::Mat blob = cv::dnn::blobFromImage(
        raw_frame, 
        1.0 / 255.0, 
        cv::Size(target_width, target_height), 
        cv::Scalar(), 
        true,  // swapRB (BGR to RGB)
        false  // crop
    );

    // Flatten the 4D cv::Mat into a 1D std::vector<float> for our zero-copy ONNX engine
    std::vector<float> input_tensor(
        (float*)blob.ptr<float>(0), 
        (float*)blob.ptr<float>(0) + blob.total()
    );

    // ==========================================
    // 4. RUN INFERENCE (With Performance Timer)
    // ==========================================
    std::cout << "[INFO] Running Inference..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<Detection> detections = engine.runInference(input_tensor);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cout << "[INFO] Inference completed in " << duration_ms << " ms. Found " 
              << detections.size() << " objects." << std::endl;

    // ==========================================
    // 5. POST-PROCESSING: DRAW BOXES
    // ==========================================
    // Calculate scaling factors to map 640x640 boxes back to original image size
    float scale_x = static_cast<float>(raw_frame.cols) / target_width;
    float scale_y = static_cast<float>(raw_frame.rows) / target_height;

    for (const auto& det : detections) 
    {
        // Rescale the bounding box
        int x = static_cast<int>(det.box.x * scale_x);
        int y = static_cast<int>(det.box.y * scale_y);
        int w = static_cast<int>(det.box.width * scale_x);
        int h = static_cast<int>(det.box.height * scale_y);

        cv::Rect scaled_box(x, y, w, h);

        // Draw the box (Green color, thickness 2)
        cv::rectangle(raw_frame, scaled_box, cv::Scalar(0, 255, 0), 2);

        // Draw the confidence score above the box
        std::string label = "Class " + std::to_string(det.class_id) + " : " + std::to_string(det.confidence).substr(0, 4);
        
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(raw_frame, cv::Point(x, y - labelSize.height - 5), 
                      cv::Point(x + labelSize.width, y), cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(raw_frame, label, cv::Point(x, y - 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    // Save the final image to disk
    std::string output_path = "output_detection.jpg";
    cv::imwrite(output_path, raw_frame);
    std::cout << "[INFO] Result saved to " << output_path << std::endl;

    return 0;
}