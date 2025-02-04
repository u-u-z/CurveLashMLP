import tensorflow as tf

model = tf.keras.models.load_model('curve_model.keras')

model.summary()

for layer in model.layers:
    print(f"Layer name: {layer.name}, type: {type(layer)}")
