import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Define the RCNN architecture
inputs = Input(shape=(128, 128, 1))
conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

flatten = Flatten()(pool4)
dense1 = Dense(128, activation='relu')(flatten)
dense2 = Dense(128, activation='relu')(dense1)
outputs = Dense(4, activation='softmax')(dense2)

# Create the model
model = Model(inputs=inputs, outputs=outputs)
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])

# Define data augmentation and normalization
data_augmentation = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

data_normalization = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True
)

# Load and preprocess the data
train_datapath =  "D:/Desktop/water leakage detection/dataset"

train_generator = data_augmentation.flow_from_directory(
    train_datapath,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=16,
    shuffle=True,
    class_mode='categorical'
)

validation_generator = data_normalization.flow_from_directory(
    train_datapath,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=16,
    shuffle=False,
    class_mode='categorical'
)

print("Number of training samples:", train_generator.samples)
print("Number of validation samples:", validation_generator.samples)

# Train the model
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=100
)

# Evaluate the model
y_pred = model.predict(validation_generator)
y_pred = tf.argmax(y_pred, axis=1).numpy()
y_true = validation_generator.classes

accuracy = accuracy_score(y_true, y_pred)
print("Model accuracy:", accuracy)

precision = precision_score(y_true, y_pred, average='macro')
print("Model precision:", precision)

recall = recall_score(y_true, y_pred, average='macro')
print("Model recall:", recall)

f1 = f1_score(y_true, y_pred, average='macro')
print("Model F1 score:", f1)
