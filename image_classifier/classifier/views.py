# classifier/views.py
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import numpy as np
import cv2
import keras
from PIL import Image
from tensorflow.keras.models import load_model

CATEGORIES = ['cats', 'dogs', 'chicken']
model = load_model('model_trained.h5')  # Load model once to avoid reloading

def image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img, (60, 60))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 60, 60, 1)
    return new_arr

def predict_image(request):
    context = {}
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(file_path)

        img_array = image(file_path)
        prediction = model.predict([img_array])
        prediction_result = CATEGORIES[prediction.argmax()]

        # Open the image to add the prediction text
        img = cv2.imread(file_path)
        img = cv2.resize(img, (300, 300))
        cv2.rectangle(img, (0, 0), (300, 40), (0, 0, 255), -1)
        cv2.putText(img, f"Prediction: {prediction_result}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.show()  # To show the image (Optional, depends on your setup)

        context['prediction'] = prediction_result
        context['image_url'] = file_url

    return render(request, 'classifier/index.html', context)
