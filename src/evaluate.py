import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

TEST_DIR = r"D:\Brain_tumor_project\dataset_split\test"

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

model = tf.keras.models.load_model(
    r"D:\Brain_tumor_project\models\brain_tumor_model.h5"
)

predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)

print("Confusion Matrix:")
print(confusion_matrix(test_data.classes, y_pred))

print("\nClassification Report:")
print(classification_report(
    test_data.classes,
    y_pred,
    target_names=list(test_data.class_indices.keys())
))