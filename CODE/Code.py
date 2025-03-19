import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Organize dataset paths
root_path = '/kaggle/input/fresh-and-stale-classification/dataset'
output_path = '/kaggle/working'

# Define ImageDataGenerator with data augmentation
generator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 80% training, 20% validation
)

# Create train and validation generators
train_generator = generator.flow_from_directory(
    os.path.join(root_path, 'Train'),
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    interpolation='bilinear'
)

test_generator = generator.flow_from_directory(
    os.path.join(root_path, 'Train'),
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    interpolation='bilinear'
)

# Display class indices
print(test_generator.class_indices)

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(test_generator.class_indices), activation="softmax")  # Ensure correct output classes
])

# Display model summary
model.summary()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=test_generator,
    callbacks=[early_stopping]
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Fetch a batch of test images
batch = next(test_generator)
X_test_batch, Y_test_batch = batch

# Predict on the batch
predictions = model.predict(X_test_batch)

# Get class labels from generator
class_indices = test_generator.class_indices
labels = {v: k for k, v in class_indices.items()}

# Display predictions
num_images = min(16, len(X_test_batch))
plt.figure(figsize=(15, 15))

for i in range(num_images):
    image = X_test_batch[i]
    predicted_class_idx = np.argmax(predictions[i])
    predicted_class = labels[predicted_class_idx]
    actual_class_idx = np.argmax(Y_test_batch[i])
    actual_class = labels[actual_class_idx]
    
    plt.subplot(4, 4, i + 1)
    plt.imshow(image)
    plt.title(f"Prediction: {predicted_class}\nActual: {actual_class}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('sv.png')
plt.show()
