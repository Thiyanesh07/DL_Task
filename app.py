
import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


model = load_model("img_classification.keras")

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def predict_image(img):
    img = img.resize((32, 32))  
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)[0]  # shape (10,)
    
    
    probs = {class_names[i]: float(pred[i]) for i in range(len(class_names))}
    return probs


interface = gr.Interface(
    fn=predict_image, 
    inputs=gr.Image(type="pil"), 
    outputs=gr.Label(num_top_classes=10),  # shows probabilities for all classes
    title="CIFAR-10 Image Classifier",
    description="Upload an image and see the predicted probabilities for all classes."
)


interface.launch()

