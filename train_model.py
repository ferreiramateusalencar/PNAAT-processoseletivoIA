import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Carregamento do dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalização e reshape para CNN
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# 2. Construção do modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.summary()

# 3. Compilação
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 4. Treinamento
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# 5. Avaliação final
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\n Acurácia final no conjunto de teste: {accuracy * 100:.2f}%")

# 6. Salvamento do modelo
model.save("model.h5")
print(" Modelo salvo em: model.h5")
