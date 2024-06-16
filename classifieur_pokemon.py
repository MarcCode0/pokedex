import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.applications import ResNet50, ResNet101
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Configuration
data_dir = r"C:\Users\leoni\Documents\Data"
# data_dir = r"C:\Users\Marc\Pictures\Pokemon\Data"
# val_dir = r"C:\Users\Marc\Pictures\Pokemon\Data"
num_classes = 20  # nombre de classes Pokémon
batch_size = 32
epochs = 20
img_height = 224
img_width = 224

############ Data Augmentation ############
# Création d'un générateur d'images avec augmentation de données
train_data_generator = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_data_generator.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),  # Taille d'entrée de ResNet50 et 101
    batch_size=batch_size,
    class_mode='sparse',
    subset='training',
    seed = 1,
    shuffle = True
)

val_data_generator = ImageDataGenerator(validation_split=0.2)

val_data = val_data_generator.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation',
    seed = 1,
    shuffle = False
)

# nom des classes
class_names = list(train_data.class_indices.keys())

### Méthode de chargement du CNN classique ###

# train_data = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=1,
#   image_size=(img_height, img_width),
#   batch_size=batch_size,
#   shuffle=True
#   )

# val_data = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="validation",
#   seed=1,
#   image_size=(img_height, img_width),
#   batch_size=batch_size,
#   shuffle=False
#   )

# nom des classes
# class_names = train_data.class_names

##############################################

# Chargement du modèle pré-entraîné ResNet101 sans la couche de classification ImageNet
base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

#Dégeler certaines couches pour le fine-tuning
# for layer in base_model.layers[:10]:
#     layer.trainable = True

# Ajout de nouvelles couches de classification adaptées aux classes Pokémon
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x) # test
x = Dense(1024)(x)
x = BatchNormalization()(x)  # Ajout de la couche BatchNormalization
x = Activation('relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = keras.models.Model(inputs=base_model.input, outputs=predictions)

# Geler les couches du modèle pré-entraîné
# for layer in base_model.layers:
#     layer.trainable = False

## Définir un  optimizer ##
# initial_learning_rate = 0.01
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=10000,
#     decay_rate=0.9,
#     staircase=True)
# opti = Adam(learning_rate=lr_schedule)

# Compilation du modèle
# avec la méthode d'origine (train_data_generator.flow_from_directory) utiliser
# loss='sparse_categorical_crossentropy'
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

logdir="logs"

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1, write_images=logdir)

history = model.fit( 
  train_data,
  validation_data=val_data,
  epochs=epochs,
  callbacks=[tensorboard_callback]
)

############### Save model ###############

# Sauvegarder le modèle
# model.save('model_resnet')

# Charger le modèle
# my_model = tf.keras.models.load_model('model_resnet101_20class')

############### Test model ###############

# Charger et prétraiter l'image
def preprocess_image(img_path, img_height, img_width):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Faire une prédiction
def predict_image(model, img_array, class_names, display):
    predictions = model.predict(img_array)
    output = np.argmax(predictions[0])
    proba = predictions[0][output]
    # Display permet d'afficher une phrase annonçant la classe prédite
    if display == 1:
        print("C'est un " + class_names[output] + " ! (" + str(round(proba*100,1)) + "%)")
    return class_names[output], proba

# img1 = preprocess_image(r"C:\Users\Marc\Documents\Test\Alakazam_2.jpg",img_height,img_width)
# predict_image(model20,img1,class_names,1)

# Tester tout un dossier
def evaluate_model(model, test_dir, class_names):
    correct = 0
    total = 0
    
    # Liste des fichiers dans le dossier de test
    test_images = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    
    for img_name in test_images:
        
        img_path = os.path.join(test_dir, img_name) 
        # Charger et prétraiter l'image        
        img_array = preprocess_image(img_path,img_height,img_width)    
        # Faire une prédiction (sans afficher de phrase avec display = 0)
        predicted_class, _ = predict_image(model, img_array, class_names, 0)              
        # Nom de la classe attendue (les deux derniers caractères et l'extension du nom de l'image sont enlevés)
        expected_class = img_name[:-6]        
        # Comparer la prédiction avec la classe attendue
        if predicted_class == expected_class:
            correct += 1
            
        total += 1    
    # Calculer le ratio de prédictions correctes
    ratio = correct / total
    return ratio

# evaluate_model(model20,r"C:\Users\Marc\Documents\Test",class_names)

############### Affichage ###############

# Données d'accuracy et de loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

# Créer les subplots
plt.figure(figsize=(12, 6))

# Training and Validation accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Training and Validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Afficher les subplots
plt.tight_layout()
plt.show()

# Afficher des images du dataset augmenté
def display_images2(data_generator, class_names):
    # Extraire un batch d'images et de labels du générateur de données
    images, labels = next(data_generator)
    plt.figure(figsize=(10, 10))
    for i in range(min(9, images.shape[0])):  # Affiche jusqu'à 9 images du batch
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
    plt.tight_layout()
    plt.show()