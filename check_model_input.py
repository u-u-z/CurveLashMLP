import tensorflow as tf

# 加载模型
model = tf.keras.models.load_model('curve_model.keras')

# 打印模型配置
print("\nModel config:")
print(model.get_config())

# 打印每一层的配置
for i, layer in enumerate(model.layers):
    print(f"\nLayer {i}: {layer.name}")
    print("Config:", layer.get_config())
