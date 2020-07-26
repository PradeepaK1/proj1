import IPython.display as ipd
import sounddevice as sd
import soundfile as sf
import streamlit as st
from pydub import AudioSegment 
import speech_recognition as sr 
from glob import glob
from scipy.io import wavfile as wav
import numpy as np
from PIL import Image
import IPython.display as ipd
from IPython.display import Audio
import librosa
import librosa.display
import pandas as pd
import os
import pyaudio
import wave
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint 
from datetime import datetime 
from sklearn.model_selection import train_test_split
ipd.Audio('notes/fold1/DA.1.wav')
filename = 'notes/fold1/DA.1.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold1/DA.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold1/ni.1.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold1/ni.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold2/NI.1.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold2/NI.2.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold2/NI.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold3/ri.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold3/SA.1.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold3/SA.2.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold3/SA.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold4/ga.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold4/RI.1.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold4/RI.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold5/GA.1.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold5/GA.2.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold5/GA.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold6/MA.1.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold6/ma.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold7/da.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold7/PA.1.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold7/PA.2.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'notes/fold7/PA.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
import pandas as pd
metadata = pd.read_csv('notes/metadata.csv')
filename = 'notes/fold5/GA.wav' 
librosa_audio, librosa_sample_rate = librosa.load(filename) 
scipy_sample_rate, scipy_audio = wav.read(filename) 
mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc=40)
def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None 
     
    return mfccsscaled 
fulldatasetpath = 'notes/'
metadata = pd.read_csv('notes/metadata.csv')
features = []
for index, row in metadata.iterrows():
    file_name = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    class_label = row["class_name"]
    data = extract_features(file_name)  
    features.append([data, class_label])
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)
num_labels = yy.shape[1]
filter_size = 2
model = Sequential()
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100*score[1]
num_epochs = 1000
num_batch_size = 32
checkpointer = ModelCheckpoint(filepath='notes/saved_models/weights.best.basic_mlp.hdf5', 
                               verbose=0, save_best_only=True)
start = datetime.now()
model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=0)
duration = datetime.now() - start        
score = model.evaluate(x_train, y_train, verbose=0)

score = model.evaluate(x_test, y_test, verbose=0)
def extract_feature(file_name):
   
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None, None

    return np.array([mfccsscaled])
def print_prediction(file_name):
    prediction_feature = extract_feature(file_name) 
    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector) 
    st.write("key:", predicted_class[0], '\n') 
    f = open('notes/log.txt', 'a')
    print(predicted_class[0], file = f)
    predicted_proba_vector = model.predict_proba(prediction_feature) 
    predicted_proba = predicted_proba_vector[0]
    return 0 
st.title("WELCOME TO THE WORLD OF MUSIC!")
vid=open("example.mp4","rb")
st.video(vid)
st.markdown("<span style=“background-color:#121922”>",unsafe_allow_html=True)
audio = st.file_uploader("Choose an audio...", type="wav")
if audio is not None:
    audio = AudioSegment.from_wav(audio)
    n = len(audio) 
    counter = 1
    interval = 1 * 1000
    overlap = 0
    start = 0
    end = 0
    flag = 0
    for i in range(0,  n, interval): 

        if i == 0: 
            start = 0
            end = interval 
        else: 
            start = end  -overlap
            end = start + interval  
        if end >= n: 
            end = n 
            flag = 1 
        chunk = audio[start:end] 
        filename = 'notes/test2/chunk'+str(counter)+'.wav'
        chunk.export(filename, format ="wav") 
        counter = counter + 1
    for name in glob('notes/test2/*.wav'):
        filename = name
        print_prediction(filename)
st.sidebar.title("Duration")
duration = st.sidebar.slider("Recording duration", 0.0, 3600.0, 3.0)
def record_and_predict(duration):
    filename = "recorded.wav"
    chunk = 1024
    FORMAT = pyaudio.paInt16
    channels = 1
    sample_rate = 44100
    record_seconds = duration
    p = pyaudio.PyAudio()
    # open stream object as input & output
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    frames = []
    print("Recording...")
    for i in range(int(44100 / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))
    wf.close()                                                            
    audio="recorded.wav"
    if audio is not None:
        audio = AudioSegment.from_wav(audio)
        n = len(audio) 
        counter = 1
        interval = 1 * 1000
        overlap = 0
        start = 0
        end = 0
        flag = 0
        for i in range(0,  n, interval): 
            if i == 0: 
                start = 0
                end = interval 
            else: 
                start = end  -overlap
                end = start + interval  
            if end >= n: 
                end = n 
                flag = 1            
            chunk = audio[start:end] 
            filename = 'notes/test2/chunk'+str(counter)+'.wav'
            chunk.export(filename, format ="wav") 
            counter = counter + 1
        for name in glob('notes/test2/*.wav'):
            filename = name
            print_prediction(filename)
if st.button("Start Recording"):
    with st.spinner("Recording..."):
        record_and_predict(duration)

