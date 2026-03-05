import tensorflow as tf
print("--- TENSORFLOW TEST ---")
print(f"TensorFlow Version: {tf.__version__}")
devices = tf.config.list_physical_devices()
print(f"Devices found: {devices}")


a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
print("Calculation Result:", tf.matmul(a, b))