import streamlit as st
import numpy as np
# import numpy as np
import pandas as pd
from PIL import Image
# import requests
from geopy.geocoders import Nominatim
from streamlit_js_eval import get_geolocation
from io import BytesIO
import time
# import logging
from audio_recorder_streamlit import audio_recorder
import matplotlib.pyplot as plt
from tensorflow import keras
import calendar
from datetime import datetime
from geopy.geocoders import Nominatim
import requests
import json
import librosa

#loading data
desc_path = "interface/data/36ebirds_description_images.csv"
data = pd.read_csv(desc_path)
df = pd.DataFrame(data)

cord_path = "interface/data/36_species_coordinates.csv"
bird_coord = pd.read_csv(cord_path)
coord = pd.DataFrame(bird_coord)

#date retrieval

currentMonth = datetime.now().month
currentDay = datetime.now().day
currentYear = datetime.now().year
currentMonthName = calendar.month_name[currentMonth]


logo_col, title_col = st.columns([1,4])
with logo_col:
    logo = Image.open('interface/data/logo.png')
    st.image(logo)
with title_col:
    st.title("ChirpID")

#chirpID description
st.write("ChirpID is a new tool to help nature lovers, avid birders, field researchers, and all curious souls to discover the birds around them. The model uses a functional model combining dense and convolutional neutral networks to classify bird species through preprocessed audio input. Simply record or upload an audio clip, and ChirpID will tell you about the bird species you are hearing!")

with st.sidebar:
    audio_bytes = audio_recorder(
        text="Record a bird",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        #icon_name="fa-solid fa-dove",
        icon_size="3x",
        sample_rate=22050
    )

    upload_audio = st.file_uploader('Or: Upload sound for classification!', type=['wav','mp3'])

#chirpID location

    user_loc = st.sidebar.text_input('Enter your location below')
    if user_loc:
        try:
            #calling the Nominatim tool
            loc = Nominatim(user_agent="GetLoc")
            getLoc = loc.geocode(user_loc)
            # printing address
            user_lat = getLoc.latitude
            user_long = getLoc.longitude
            st.sidebar.write('Your Location is: ', getLoc.address)
            st.sidebar.write(f"({user_lat}, {user_long})")

        except:
            st.write("chirp error")
    elif st.checkbox("Use my current location"):
        loc = get_geolocation()
        try:
            user_lat = loc.get('coords').get('latitude')
            user_long = loc.get('coords').get('longitude')
            st.sidebar.write(f"Your current location is:")
            st.sidebar.write(f"({user_lat}, {user_long})")
        except:
            st.success('Loading coordinates, please wait...')

    # if audio_bytes:
    #     st.write("WE GOT AUDIO")

def preprocess_picture(file):

    sampling_rate=22050
    Signal , sr = librosa.load(file,sr=sampling_rate)

    n_fft = 512 # the window, or kernel as I understand it
    hop_length = 512 # the amount of shifting the window to the right
    stft = librosa.core.stft(Signal , hop_length = hop_length , n_fft = n_fft)
    spectogram = np.abs(stft)
    picture = librosa.amplitude_to_db(spectogram)
    return picture

def get_image_url(predicted_species):
    return df[df['en']==predicted_species]['image_links'].squeeze()

#Define a function that retrieves the description of the brid
def get_description(predicted_species):
    return df[df['en']==predicted_species]['descriptions'].squeeze()

def get_map_data(predicted_species):
    df= coord[coord['en']==predicted_species][['lat','lng']].dropna().rename(columns={'lng':'lon'})
    pd.to_numeric(df['lat'])
    pd.to_numeric(df['lon'])
    return df


if upload_audio:
    with open("audio.wav", mode="bw") as f:
        f.write(upload_audio.read())

    log_spectogram = preprocess_picture("audio.wav")
    st.audio("audio.wav", format="audio/wav")

if audio_bytes:
    abs_fp = "audio.wav"
    with open(abs_fp, mode="bw") as f:
        f.write(audio_bytes)

    log_spectogram = preprocess_picture("audio.wav")
    st.audio("audio.wav", format="audio/wav")

elif audio_bytes:
    log_spectogram = preprocess_picture("audio.wav")
    st.audio(audio_bytes, format="audio/wav")


if upload_audio or audio_bytes:
    fig, ax = plt.subplots(figsize=(15,2))
    ax.imshow(log_spectogram, extent=[0, 16, -1.2, 1.2], cmap='viridis')
    plt.xlabel('Time (s)', fontdict={'size':15})
    plt.ylabel('Frequency (kHz)', fontdict={'size':15})
    fig.savefig('spectrogram.png', bbox_inches='tight', transparent=True)

    img = Image.open('spectrogram.png')
    st.image(img)

    lng = user_long
    lat = user_lat
    month = currentMonth

    url = 'https://chirpapi-niuue56uea-an.a.run.app/predict?lng={}&lat={}&month={}'.format(lng, lat, month)

    headers = {
        'accept': 'application/json',
    }

    fp = "audio.wav"
    files = {
        'file': (fp, open(fp, 'rb')),
    }

    response = requests.post(url, headers=headers, files=files)
    prediction = json.loads(response.content)
    predicted_species = prediction["prediction"]

            # st.write(f"Our model predicts it's a {predicted_species}")

    st.header(predicted_species + " "+ df[df['en']==predicted_species]['cn'].squeeze())
    left_column, right_column = st.columns(2)

    with left_column:
        # Get the URL of the selected image
        image_url = get_image_url(predicted_species)
        st.image(image_url, caption=predicted_species)

    with right_column:
        st.write(get_description(predicted_species))

    df = get_map_data(predicted_species)
    st.map(df)
# except:
#     st.write("")
