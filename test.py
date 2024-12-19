import numpy as np
import cv2
import keras
import matplotlib.pyplot as plt

# Bước 1: Tải lại mô hình đã huấn luyện
model = keras.models.load_model('vgg16_transferlearning.h5')

name_animal = ['squirrel', 'horse', 'butterfly', 'cow', 'cat', 'sheep', 'chicken', 'elephant', 'spider', 'dog']

# Bước 2: Xử lý ảnh mới
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Chỉnh sửa kích thước ảnh cho phù hợp với mô hình VGG16 (224x224)
    if img.shape[0] > img.shape[1]:
        tile_size = (int(img.shape[1] * 256 / img.shape[0]), 256)
    else:
        tile_size = (256, int(img.shape[0] * 256 / img.shape[1]))

    img = centering_image(cv2.resize(img, dsize=tile_size))

    # Cắt ra khu vực trung tâm (224x224)
    img = img[16:240, 16:240]
    img = img.astype('float32') / 255  # Chuẩn hóa ảnh
    return img

# Bước 3: Dự đoán một ảnh mới
def predict_image(image_path):
    img = preprocess_image(image_path)  # Tiền xử lý ảnh
    img = np.expand_dims(img, axis=0)  # Thêm batch dimension (1, 224, 224, 3)

    prediction = model.predict(img)  # Dự đoán
    predicted_class = np.argmax(prediction, axis=1)[0]  # Lấy lớp dự đoán có xác suất cao nhất

    return predicted_class


def centering_image(img):
    size = [256, 256]  # Kích thước hình ảnh cuối cùng
    img_size = img.shape[:2]
    
    # Căn giữa hình ảnh
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img
    
    return resized

# Bước 4: Hiển thị kết quả
def show_result(image_path, predicted_class):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(f'Predicted: {name_animal[predicted_class]}')  # Hiển thị tên động vật
    plt.axis('off')
    plt.show()

# Ví dụ sử dụng: Dự đoán một ảnh từ tập test
image_path = 'cat.jpg'  # Thay bằng đường dẫn ảnh của bạn
predicted_class = predict_image(image_path)
show_result(image_path, predicted_class)
