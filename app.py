import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

#------------------------------------------------------------------------------------------------------------

import streamlit as st
import base64

# Function to set background from local image file
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    st.markdown(f"""
         <style>
         .stApp {{
             background-image: url("data:image/jpg;base64,{encoded_string}");
             background-size: cover;
             background-position: center;
             background-repeat: no-repeat;
             background-attachment: fixed;
          
         }}
         </style>
         """, unsafe_allow_html=True)

# Call the function with your file
add_bg_from_local("cybersecurity-concept-collage-design.jpg")

#---------------------------------------------------------------------------------------------------------
# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to preprocess and transform input text
def transform_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    return " ".join(stemmed_words)

# Load pre-trained model and vectorizer
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Custom CSS for background and content box styling
st.markdown("""
    <style>
        .stApp {
            background-color:#25aedb;
        }
        .main-box {
            border: 2px solid #53227a;
            border-radius: 10px;
            padding: 20px;
            background-color: #ffffff;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app layout
st.title("Email/SMS Classifier")

input_sms = st.text_input("Enter the message To Check Spam or Not ")

if st.button("Predict"):
    # Preprocess input
    transformed_sms = transform_text(input_sms)

    # Vectorize input
    vector_input = tfidf.transform([transformed_sms])

    # Predict using the model
    result = model.predict(vector_input)

    # Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not a Spam")


#import streamlit as st
#from PIL import Image

#image = Image.open("cybersecurity-concept-collage-design.jpg")  # Replace with your file name
#st.image(image, caption='Your Image Caption', use_column_width=True)
