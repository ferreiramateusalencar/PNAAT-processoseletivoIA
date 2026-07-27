import tensorflow as tf
import os

#insira seu código aqui
model = tf.keras.models.load_model("model.h5")
print("Modelo carregado: model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()


with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo otimizado salvo em: model.tflite")

h5_size     = os.path.getsize("model.h5")     / 1024
tflite_size = os.path.getsize("model.tflite") / 1024

print(f"\nTamanho do modelo original (.h5):      {h5_size:.1f} KB")
print(f"Tamanho do modelo otimizado (.tflite): {tflite_size:.1f} KB")
print(f"Redução de tamanho: {((h5_size - tflite_size) / h5_size * 100):.1f}%")
