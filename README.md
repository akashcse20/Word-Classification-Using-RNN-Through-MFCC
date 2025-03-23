# Word Classification Using RNN Through MFCC

This repository demonstrates the use of **Mel-Frequency Cepstral Coefficients (MFCC)** for extracting features from audio data and **Bidirectional LSTM (Long Short-Term Memory)** for classifying words in speech. The primary goal of the project is to recognize spoken words by training a model on audio data, extracting relevant features, and evaluating the performance of the classifier using common metrics such as accuracy, precision, recall, and F1-score.

---

## 🛠️ Key Features

- **MFCC Feature Extraction**: Converts raw audio data into MFCC features, which are commonly used for speech and audio classification.
- **Bidirectional LSTM**: A **Bidirectional LSTM** model is utilized to capture the dependencies of sequential data both from the past and future, enhancing classification accuracy.
- **Data Preprocessing**: Data preprocessing includes reading and cleaning MFCC files, encoding labels, and splitting the data for training and testing.
- **Model Evaluation**: Evaluation metrics like **accuracy**, **precision**, **recall**, and **F1-score** provide a comprehensive understanding of the model's performance.
- **Confusion Matrix**: A **confusion matrix** is generated to visualize the performance and identify misclassifications among different classes.

---

## ⚙️ Prerequisites

To run this project, you will need the following Python packages:

```bash
pip install librosa tensorflow sklearn matplotlib seaborn
```

Ensure that you have a Python environment set up with **TensorFlow** for model training, **Librosa** for audio feature extraction, and **scikit-learn** for machine learning tasks.

---

## 📂 Project Structure

The project directory contains the following structure:

```
/Word-Classification-Using-RNN-Through-MFCC
├── /data
│   └── MFCC Dataset
├── /model
│   └── Trained Model (saved as .h5 file)
├── /notebooks
│   └── Jupyter notebooks for data exploration and training
├── README.md
└── LICENSE
```

---

## 🧑‍💻 Setup Instructions

### 1. Mount Google Drive

In order to access your dataset stored in Google Drive, mount the drive to your environment:

```python
from google.colab import drive
drive.mount('/content/drive')
```

This allows you to work with datasets directly from your Google Drive, ensuring your files are accessible for data processing and training.

---

### 2. Import Libraries

You'll need to import several libraries to handle data processing, model creation, training, and evaluation:

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping
```

These libraries are essential for working with audio features, building machine learning models, and visualizing results.

---

## 📊 Data Visualization

Before training the model, it is useful to visualize the MFCC features to understand the data better. This helps in ensuring that the features are well-defined for classification.

### Plotting MFCC Features

Use the following function to visualize the MFCC features for a given dataset sample:

![Image](https://github.com/user-attachments/assets/6993a15e-7193-486f-979a-357bd6366af8)

This function generates a plot for the given **MFCC sample**, showing how the feature data behaves over time.

---

## 📂 Data Preprocessing

### Reading MFCC Data

The **MFCC features** are stored in files, where each file represents a spoken word. These features are extracted from the audio samples in the **.MFC** format, which can be loaded using Python.

```python
# Initialize empty lists for data and labels
data = []
labels = []

# Iterate over each class (word) directory
for class_folder in os.listdir(dataset_dir):
    class_folder_path = os.path.join(dataset_dir, class_folder)
    
    # Ensure it's a directory
    if os.path.isdir(class_folder_path):
        for file_name in os.listdir(class_folder_path):
            if file_name.endswith('.MFC'):
                file_path = os.path.join(class_folder_path, file_name)
                
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    data_values = [float(line.strip()) for line in lines[1:]]
                    data.append(data_values)
                    labels.append(class_folder)

data = np.array(data)
labels = np.array(labels)
```

This script reads **MFCC files**, extracts the feature values, and assigns them the correct class label based on the folder name.

---

### Label Encoding

In classification tasks, it’s necessary to convert categorical labels into numerical values for compatibility with machine learning algorithms. This can be done using **LabelEncoder** from **scikit-learn**:

```python
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
```

This converts the string labels (such as 'word1', 'word2') into integer labels (0, 1, 2, ...).

---

### Data Splitting

Once the data and labels are processed, split them into training and testing sets for model training and evaluation:

```python
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
```

This step divides the data into an 80-20 train-test split, ensuring that the model has sufficient data to train while leaving a portion for evaluation.

---

## 🧠 Model Architecture

### Building the Bidirectional LSTM Model

A **Bidirectional LSTM** model is ideal for sequential data like speech, as it can capture information from both past and future sequences. The architecture of the model includes:

1. **Bidirectional LSTM Layer**: Captures temporal dependencies from both directions.
2. **Dropout Layer**: Reduces overfitting by randomly setting a fraction of input units to 0 during training.
3. **Dense Layer**: Outputs predictions for each class.

```python
model = tf.keras.Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    Bidirectional(LSTM(128)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(set(labels)), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

This model is trained on the **MFCC features** and is capable of recognizing words based on the sequential data it processes.

---

## 🎯 Training the Model

### Early Stopping

**Early stopping** is used to halt training if the model’s validation loss stops improving for a set number of epochs:

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
```

This prevents the model from overfitting by stopping training at the optimal point.

### Training the Model

The model is trained using the **Adam optimizer** with a **learning rate** of 0.001 and the **sparse categorical cross-entropy loss** function:

```python
history = model.fit(X_train[:, :, np.newaxis], y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
```

The training process runs for up to **50 epochs** with an early stopping mechanism, monitoring the **validation loss**.

---

## 📈 Model Evaluation

This script calculates the **precision**, **recall**, and **F1-score**, and generates a detailed **classification report**.

```
Accuracy: 93.33%
Precision: 95.38%
Recall: 93.33%
F1 Score: 93.52%

**Classification Report:**
```
```
                 precision    recall  f1-score   support

     class_A       1.00      1.00      1.00         2
    class_AA       1.00      1.00      1.00         5
     class_E       1.00      1.00      1.00         5
    class_KA       1.00      0.75      0.86        12
    class_MO       1.00      1.00      1.00         5
     class_O       1.00      0.86      0.92         7
    class_OA       1.00      1.00      1.00         3
   class_SHA       1.00      1.00      1.00         6
    class_TA       0.69      1.00      0.82         9
     class_U       1.00      1.00      1.00         6

        accuracy                           0.93        60
       macro avg       0.97      0.96      0.96        60   
    weighted avg       0.95      0.93      0.94        60
```

### Confusion Matrix

A **confusion matrix** is a great tool to visualize how well the model performs on each class:

![Image](https://github.com/user-attachments/assets/7641f423-42e4-462e-9452-30b3aa700669)

This visualizes how many predictions were correct for each class and where the model made errors.

---

### 📜 License & References

- **MIT License**: This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
- **Librosa**: For extracting **MFCC features** from

 audio files.
- **TensorFlow**: For building and training the model.
- **scikit-learn**: For data preprocessing and evaluation.

---
