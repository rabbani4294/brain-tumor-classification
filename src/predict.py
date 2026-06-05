import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

model = tf.keras.models.load_model(
    r"D:\Brain_tumor_project\models\brain_tumor_model.h5"
)

# ⚠️ MUST MATCH TRAINING ORDER EXACTLY
classes = ["glioma", "meningioma", "no_tumor", "pituitary"]

def predict_tumor(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # correct preprocessing
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    print("\nPrediction:", classes[idx])
    print("Confidence:", round(confidence, 2))

predict_tumor(r"D:\Brain_tumor_project\dataset_split\test\meningioma\m3 (69).jpg")