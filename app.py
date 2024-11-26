import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model_1.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our goal is to assist you in identifying plant diseases swiftly and accurately. Upload a plant image, and our system will analyze it for any signs of disease. Let's work together to protect crops and ensure a healthy harvest!

    ### How It Works
    1. **Upload Image:** Navigate to the **Disease Recognition** page and upload an image of a plant you suspect may be diseased.
    2. **Analysis:** Our system will examine the image to detect potential diseases.
    3. **Results:** Get quick results along with recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system leverages cutting-edge machine learning for precise disease identification.
    - **User-Friendly Design:** Enjoy a smooth and intuitive interface for a great user experience.
    - **Fast Processing:** Get results within seconds to enable prompt decisions.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar, upload an image, and see the Plant Disease Detection System in action!

    ### About Us
    Find out more about our project, team, and goals on the About page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_container_width=True)
    #Predict button
    if(st.button("Predict")):
        if test_image is not None:
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            #Reading Labels
            class_name = ['Apple_scab', 'Apple_Black_rot', 'Cedar_apple_rust', 'Apple_healthy',
                    'Blueberry_healthy', 'Cherry_Powdery_mildew', 
                    'Cherry_healthy', 'Corn_(maize)_Cercospora_leaf_spot', 
                    'Corn_(maize)__Common_rust', 'Corn_(maize)_Northern_Leaf_Blight', 'Corn_(maize)_healthy', 
                    'Grape_Black_rot', 'Grape_Esca_(Black_Measles)', 'Grape_Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape_healthy', 'Orange_Haunglongbing_(Citrus_greening)', 'Peach_Bacterial_spot',
                    'Peach_healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 
                    'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy', 
                    'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 
                    'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 
                    'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
                    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites', 
                    'Tomato_Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus',
                      'Tomato_healthy']
            st.success("Model is Predicting it's a {}".format(class_name[result_index]))
        else:
            st.warning("Please upload an image first!")