import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical

# Các lớp động vật (tương ứng với các chỉ số lớp trong mô hình)
animals = ['squirrel', 'horse', 'butterfly', 'cow', 'cat', 'sheep', 'chicken', 'elephant', 'spider', 'dog']

# Tải mô hình đã huấn luyện (giả sử mô hình đã được lưu ở dạng file .model)
model = load_model('VGG16-transferlearning.model')

# Hàm tiền xử lý ảnh đầu vào
def preprocess_image(image_path):
    # Đọc ảnh và chuyển đổi màu từ BGR (OpenCV) sang RGB
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Đảm bảo rằng ảnh có kích thước 224x224
    img = cv2.resize(img, (224, 224))

    # Tiền xử lý ảnh cho VGG16
    img_array = image.img_to_array(img)  # Chuyển đổi ảnh thành mảng numpy
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
    img_array = preprocess_input(img_array)  # Tiền xử lý cho VGG16

    return img_array

# Hàm dự đoán động vật
def predict_animal(image_path):
    # Tiền xử lý ảnh
    img_array = preprocess_image(image_path)
    
    # Dự đoán với mô hình
    predictions = model.predict(img_array)
    
    # Lấy chỉ mục của lớp có xác suất cao nhất
    predicted_class = np.argmax(predictions)
    
    return predicted_class, predictions[0][predicted_class]

# Hàm hiển thị ảnh và kết quả dự đoán
def display_image_and_prediction(image_path):
    # Dự đoán kết quả
    predicted_class, probability = predict_animal(image_path)
    
    # Hiển thị ảnh
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')  # Tắt trục
    plt.show()

    # In ra dự đoán
    print(f"Predicted Animal: {animals[predicted_class]}")
    print(f"Probability: {probability * 100:.2f}%")

# Test chương trình với một ảnh đầu vào
image_path = 'path_to_your_image.jpg'  # Thay đổi đường dẫn đến ảnh của bạn
display_image_and_prediction(image_path)
