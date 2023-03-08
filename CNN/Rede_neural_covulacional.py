# Importando as bibliotecas necessárias
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Carregando o conjunto de dados MNIST (dígitos manuscritos)
(train_images, train_labels), (test_images,
                               test_labels) = datasets.mnist.load_data()

# Normalizando as imagens de entrada para que seus valores fiquem entre 0 e 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Definindo o modelo da CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compilando o modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

# Treinando o modelo
model.fit(train_images.reshape(-1, 28, 28, 1), train_labels, epochs=5,
          validation_data=(test_images.reshape(-1, 28, 28, 1), test_labels))

# Avaliando o modelo com os dados de teste
test_loss, test_acc = model.evaluate(
    test_images.reshape(-1, 28, 28, 1),  test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')
