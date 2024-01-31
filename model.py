import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

np.set_printoptions(suppress=True)
# Tải model
model = load_model("converted_keras/keras_model.h5", compile=False)
class_names = open("converted_keras/labels.txt", "r").readlines()

# Giao diện Streamlit
st.title("Dự đoán sâu bệnh trên cây Sầu riêng")

# Tải ảnh từ người dùng
uploaded_image = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])

# labels = ["Cháy lá", "Nấm hồng", "Nhện đỏ", "Rầy phấn", "Rệp sáp", "Sâu đục thân", "Sâu đục trái", "Không bệnh"]


# def preprocess_image(img):
#     img = img.resize((224, 224))
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     return img_array
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

if uploaded_image is not None:
    # Hiển thị ảnh
    st.image(uploaded_image, caption= "Uploaded Image.", width=250)

    # Chuyển đổi ảnh thành dữ liệu numpy array
    image = Image.open(uploaded_image).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    # # # Thực hiện dự đoán
    # processed_image = preprocess_image(image)
    # prediction = model.predict(processed_image)
    # predicted_class = np.argmax(prediction)
    # predicted_label = class_names[2:]

    # # Hiển thị kết quả
    # st.write("Dự đoán:", predicted_label)
    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    st.write("Dự đoán bệnh:")
    st.title(class_name[2:])
    #print("Confidence Score:", confidence_score)
