'''
!/usr/bin/env_tf
v 0.1.0
@author:ayanava_dutta
-*-coding:utf-8-*-
'''

import streamlit as st
import base64
import pandas as pandas
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import Image, display
import warnings
from io import BytesIO
from PIL import Image, ImageOps
st.set_option('deprecation.showPyplotGlobalUse', False)

page_bg_img = '''
    <style>
    body {
    background-image: url("https://drive.google.com/file/d/1YyJ8-lYULOW8-xw-us15Y2fQRyUbkvfs/view?usp=sharing");
    background-size: cover;
    }
    </style>
    '''
st.markdown(page_bg_img, unsafe_allow_html=True)



def main():

    st.title("Covid-19 Detection using Radiography images")

    war=st.checkbox("Show Warning",value=True)
    if war:
        st.warning("Note: This app is made as a part of research work only")


    uploaded_img_file = st.file_uploader("Upload your image", type=['png'])
    
  

    if uploaded_img_file is not None:
        col1, col2, col3 = st.beta_columns([1,1,1])
        if col2.button("Predict"):
            with st.spinner('Generating Predictions...'): 
                byte_file = uploaded_img_file.read()
                with open('img.png', mode='wb+') as f:
                    f.write(byte_file)
                #image_file = BytesIO(uploaded_img_file.getvalue().decode("utf-8"))
                #image_file=uploaded_img_file.read()
                image_prediction_and_visualization('img.png',last_conv_layer_name = "conv5_block3_3_conv")


    else:
        st.warning("Please upload image file to continue")


        

def _get_model():
    
    model = load_model("bestmodel.h5")
    return model

def get_img_array(img_path):
    path = img_path
    img = image.load_img(path, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img , axis= 0 )
    return img

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
 
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

   
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

   
    grads = tape.gradient(class_channel, last_conv_layer_output)

    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()



def save_and_display_gradcam(img_path , heatmap, cam_path="cam.jpg", alpha=0.4):
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    return cam_path
    

def image_prediction_and_visualization(path,last_conv_layer_name = "conv5_block3_3_conv"):
    model = _get_model()
    img_array = get_img_array(path)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    
    img = get_img_array(path)
    class_type = {0:'Covid',  1 : 'Normal'}
    res = class_type[np.argmax(model.predict(img))]
    st.subheader("Summary:")
    l1, l2 = st.beta_columns(2)
    l1.write(f"The given image is of type : {res}")
    l2.write("The chances of Covid infected :" +str(model.predict(img)[0][0]*100)[:5]+ "%")
    l2.write("The chances of being Normal :"+str( (100 - model.predict(img)[0][1]*100)[:5])+ "%")

    with st.spinner("Generating results....."):
        grad_img=save_and_display_gradcam(path, heatmap)
   
    
    c1, c2 = st.beta_columns(2)

    c1.image(path,caption='Original image')

    c2.image(grad_img,caption='Image with weighted region of interest')
   
    


if __name__ == '__main__':

    st.sidebar.markdown(
    """
    <h1><center>COVID Detection Tool</center></h1>
    <a style='display: block; text-align: center;' target="_blank" href="https://drive.google.com/drive/folders/1xfEaAgSBEuxoo8rVwIu8_yT6QGLI2Ftg?usp=sharing">Sample images</a>
    <a style='display: block; text-align: center;' target="_blank" href="https://drive.google.com/file/d/1G2LgvUbT4O8kDgk41y-fARvQp-AXY1xn/view?usp=sharing">Working Video</a>

    """, unsafe_allow_html=True)

    
    main()
