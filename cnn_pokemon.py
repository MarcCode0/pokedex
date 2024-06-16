import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf

# data_dir = r"C:\Users\Marc\Pictures\Pokemon\Data"
# val_dir = r"C:\Users\Marc\Pictures\Pokemon\Data"

data_dir = r"C:\Users\leoni\Documents\Data2"
val_dir = r"C:\Users\leoni\Documents\Data2"

batch_size = 8
img_height = 256
img_width = 256

train_data = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=1,
  image_size=(img_height, img_width),
  batch_size=batch_size,
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
  val_dir,
  validation_split=0.2,
  subset="validation",
  seed=1,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# nom des classes
class_names = train_data.class_names

# modèle
num_classes = len(class_names)

#model = tf.keras.Sequential([
#    layers.experimental.preprocessing.Rescaling(1./255),
#    layers.Conv2D(128,4, activation='relu'),
#    layers.MaxPooling2D(),
#    layers.Conv2D(64,4, activation='relu'),
#    layers.MaxPooling2D(),
#    layers.Conv2D(32,4, activation='relu'),
#    layers.MaxPooling2D(),
#    layers.Conv2D(16,4, activation='relu'),
#    layers.MaxPooling2D(),
#    layers.Flatten(),
#    layers.Dense(64,activation='relu'),
#    layers.Dense(num_classes, activation='softmax')
#])

#model = tf.keras.Sequential([
#    layers.experimental.preprocessing.Rescaling(1./255),
#    layers.Conv2D(8, 3, activation='relu', padding='same'),
#    layers.BatchNormalization(),
#    layers.MaxPooling2D(),
#    
#    layers.Conv2D(16, 3, activation='relu', padding='same'),
#    layers.BatchNormalization(),
#    layers.MaxPooling2D(),
#    
#    layers.Conv2D(32, 3, activation='relu', padding='same'), 
#    layers.BatchNormalization(),
#    layers.MaxPooling2D(),
#    
#    layers.Conv2D(64, 3, activation='relu', padding='same'),
#    layers.BatchNormalization(),
#    layers.MaxPooling2D(),
#    
#    layers.Conv2D(128, 3, activation='relu', padding='same'),
#    layers.BatchNormalization(),
#    layers.MaxPooling2D(),
#    
#    layers.Conv2D(256, 3, activation='relu', padding='same'),
#    layers.BatchNormalization(),
#    layers.MaxPooling2D(),
#    
#    layers.Flatten(),
#    layers.Dropout(0.5),
#    layers.Dense(128, activation='relu'),
#    layers.BatchNormalization(),
#    
#    layers.Dropout(0.5),
#    layers.Dense(num_classes, activation='softmax')
#])

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    # layers.Conv2D(256, 3, activation='relu', padding='same'),
    # layers.BatchNormalization(),
    # layers.MaxPooling2D(),
    
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, activation='relu', padding='same'), 
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    
    layers.Conv2D(16, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    
    layers.Conv2D(8, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
  metrics=['accuracy'],)

logdir="logs"

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1, write_images=logdir)

history = model.fit( 
  train_data,
  validation_data=val_data,
  epochs=20,
  callbacks=[tensorboard_callback]
)

############### Affichage ###############

# Données d'accuracy et de loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

# Afficher les courbes d'accuracy
plt.figure(figsize=(8, 6))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Afficher les courbes de loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Afficher des images du dataset d'entraînement
def display_images(dataset, class_names):
    images, labels = next(iter(train_data))
    plt.figure(figsize=(10, 10))
    for i in range(min(9, images.shape[0])):
        plt.subplot(3, 3, i + 1)  # Affiche jusqu'à 9 images du batch
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()