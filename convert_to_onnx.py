import os
import tensorflow as tf
import tf2onnx
import onnx

def convert_model():
    input_path = "model/xception_deepfake.h5"
    output_path = "model/deepfake.onnx"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print("Loading Keras model...")
    model = tf.keras.models.load_model(input_path)
    
    print("Converting to ONNX...")
    # Convert using tf2onnx
    spec = (tf.TensorSpec((None, 299, 299, 3), tf.float32, name="input"),)
    output_path = "model/deepfake.onnx"
    
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    convert_model()
