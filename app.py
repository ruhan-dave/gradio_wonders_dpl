import os
from fastai.vision.all import *
from fastai.vision.widgets import *
import fastai
from fastai.vision.core import *
from fastai.learner import *
import gradio as gr


filename = f"./wonder_app_model.pkl"
learn = load_learner(fname=filename)

def predict_image(img):
    prediction = learn.predict(PILImage.create(img))
    classes = learn.dls.vocab
    probs_list = prediction[2].numpy()
    return {c: f"{round(float(probs_list[i]), 4)}" for (i, c) in enumerate(classes)}  

image = gr.inputs.Image(shape=(200,200))
label = gr.outputs.Label(num_top_classes=7)

interface = gr.Interface(
    fn=predict_image,
    inputs=image,
    outputs=label
)

interface.launch(debug=True)
