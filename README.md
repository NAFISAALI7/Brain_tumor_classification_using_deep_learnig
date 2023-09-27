# Brain_tumor_classification_using_deep_learnig

The provided code is for building and training a Convolutional Neural Network (CNN) model to detect brain tumors from medical images. Here's a step-by-step explanation of how the code works:

1. Importing necessary libraries:
   - TensorFlow: A popular deep learning framework.
   - Various components from Keras, a high-level deep learning API that runs on top of TensorFlow.
   - ImageDataGenerator: Used for data augmentation and normalization.
   - Scikit-learn metrics: Used to calculate accuracy, precision, recall, and F1-score.

2. Define the CNN architecture:
   - The model starts with an input layer of size (128, 128, 1), indicating grayscale images with a size of 128x128 pixels.
   - It consists of four convolutional layers with increasing filter sizes (32, 64, 128, and 256), each followed by a ReLU activation function and max-pooling to reduce spatial dimensions.
   - The convolutional layers help the model learn hierarchical features from the input images.
   - The fully connected layers consist of two dense layers with 128 units and ReLU activation functions.
   - The output layer has four units with a softmax activation, representing the classes of brain tumor detection (likely indicating different types of tumors).

3. Create the model:
   - The `Model` class from Keras is used to define the model, taking the input and output layers.

4. Compile the model:
   - The model is compiled using categorical cross-entropy as the loss function, Adam optimizer with a learning rate of 1e-4, and accuracy as the evaluation metric.

5. Define data augmentation and normalization:
   - Data augmentation helps increase the diversity of training data by applying random transformations (rotation, shifting, flipping) to input images.
   - Data normalization standardizes pixel values to have zero mean and unit variance, which can help improve training.

6. Load and preprocess the data:
   - The code specifies the path to the training data directory (`train_datapath`).
   - Two generators are created using `ImageDataGenerator`. The first one (`train_generator`) applies data augmentation to the training dataset, while the second one (`validation_generator`) applies data normalization to the validation dataset.
   - Images are loaded from the directory, resized to (128, 128) pixels, converted to grayscale, and organized into batches for training and validation.

7. Training the model:
   - The `model.fit()` function is used to train the model. It takes the training generator as input, along with the validation generator, and trains the model for 100 epochs.

8. Evaluate the model:
   - After training, the model is evaluated on the validation data.
   - Predictions are obtained using the trained model on the validation generator.
   - The predictions are converted into class labels using `argmax`.
   - Various performance metrics like accuracy, precision, recall, and F1-score are computed using scikit-learn functions and printed.

In summary, this code defines a CNN model for brain tumor detection, trains it using data augmentation and normalization, and evaluates its performance using standard classification metrics. This is a common workflow for developing and assessing deep learning models for medical image analysis tasks.
