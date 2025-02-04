import tensorflow as tf

# 加载模型
model = tf.keras.models.load_model('curve_model.keras')

# 打印模型结构
model.summary()

# 打印每一层的名称
for layer in model.layers:
    print(f"Layer name: {layer.name}, type: {type(layer)}")
