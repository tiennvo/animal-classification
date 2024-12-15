import numpy as np
import cv2
import keras
import streamlit as st
from PIL import Image

CATEGORIES = ['Mèo', 'Chó', 'Gà']

model = keras.models.load_model('model_trained.h5')

audio_files = {
    'Mèo': 'cat.mp3',
    'Chó': 'dog.mp3',
    'Gà': 'chicken.mp3'
}

# Function to process image
def image_to_array(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (60, 60))
    img = np.array(img)
    img = img.reshape(-1, 60, 60, 1)
    return img

st.title("Phân loại động vật")

st.write("Upload an image to classify it into cat, dog, or chicken.")

# File uploade
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
col1, col2 = st.columns([1, 1])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_array = image_to_array(image)
    
    prediction = model.predict([img_array])
    predicted_label = CATEGORIES[prediction.argmax()]


    with col2:
        st.subheader("Mô tả chi tiết")
        if predicted_label == 'Mèo':
            st.write("**Mèo**: Con mèo là một loài động vật nhỏ, lông mềm và thường được nuôi làm thú cưng. Mèo có đôi mắt sắc bén và tai nhọn. Chúng thích sống độc lập nhưng cũng rất tình cảm với chủ. Mèo thích cọ vào người, kêu meo meo và thích ngủ nhiều. Chúng cũng rất giỏi bắt chuột và chơi vờn đồ vật.")
        elif predicted_label == 'Chó':
            st.write("**Chó**: Con chó là một loài động vật nuôi rất phổ biến, thường được gọi là bạn trung thành của con người. Chó có kích thước và hình dáng đa dạng, từ nhỏ như chihuahua đến lớn như doberman. Chúng có bộ lông ngắn hoặc dài, và đôi tai có thể đứng hoặc cụp tùy giống.")
        elif predicted_label == 'Gà':
            st.write("**Gà**: Con gà là một loài gia cầm, thường được nuôi để lấy thịt và trứng. Gà có kích thước vừa phải, với bộ lông mượt và đa dạng màu sắc. Gà trống thường có bộ lông sặc sỡ và chiếc mào đỏ trên đầu, trong khi gà mái có màu sắc nhẹ nhàng hơn.")

    # tên
    with col1:
        st.write(f"Đây là con: **{predicted_label}**")
        # âm thanh
        if st.button("🔊 Play Sound"):
            # Play the audio file corresponding to the predicted animal
            audio_file = f"sound/{audio_files[predicted_label]}"
            audio_bytes = open(audio_file, 'rb').read()
            st.audio(audio_bytes, format='audio/mp3')

    # Optionally, display the image with a rectangle and label (using OpenCV)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (300, 300))
    cv2.rectangle(img, (0, 0), (300, 40), (0, 0, 255), -1)
    cv2.putText(img, f"Prediction: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image
