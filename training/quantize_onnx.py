import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "model")
    input_model = os.path.join(model_dir, "distilbert_moderation.onnx")
    output_model = os.path.join(model_dir, "distilbert_moderation_quantized.onnx")

    print(f"Loading ONNX model from: {input_model}")
    
    # Apply dynamic quantization
    quantize_dynamic(
        model_input=input_model,
        model_output=output_model,
        weight_type=QuantType.QUInt8
    )

    # Size comparison
    original_size = os.path.getsize(input_model) / (1024 * 1024)
    quantized_size = os.path.getsize(output_model) / (1024 * 1024)

    print(f"Original model size:  {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Saved quantized model to: {output_model}")

if __name__ == "__main__":
    main()
