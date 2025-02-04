import tensorflow as tf

model = tf.keras.models.load_model('curve_model.keras')

print("\nModel config:")
print(model.get_config())

for i, layer in enumerate(model.layers):
    print(f"\nLayer {i}: {layer.name}")
    print("Config:", layer.get_config())
