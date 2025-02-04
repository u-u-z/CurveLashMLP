import tensorflow as tf
import tf2onnx
import onnx

original_model = tf.keras.models.load_model('curve_model.keras')

inputs = tf.keras.Input(shape=(1,), name='model_input')
x = original_model.layers[0](inputs)
for layer in original_model.layers[1:-1]:
    x = layer(x)
outputs = original_model.layers[-1](x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='curve_model')

spec = (tf.TensorSpec((None, 1), tf.float32, name="model_input"),)
output_path = "curve_model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
print(f"Saved ONNX model to {output_path}")
