import streamlit as st
import speech_recognition as sr
import sys
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

print(sys.executable)
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load the text file and preprocess the data
with open('corona.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')

# Tokenize the text into sentences
sentences = sent_tokenize(data)

# Define a function to preprocess each sentence
def preprocess(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    # Remove stopwords and punctuation
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Preprocess each sentence in the text
corpus = [preprocess(sentence) for sentence in sentences]

# Define a function to find the most relevant sentence given a query
def get_most_relevant_sentence(query):
    # Preprocess the query
    query = preprocess(query)
    # Compute the similarity between the query and each sentence in the text
    max_similarity = 0
    most_relevant_sentence = ""
    for sentence in corpus:
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)
    return most_relevant_sentence

def transcribe_speech(audio_data):
    # Initialize the recognizer
    r = sr.Recognizer()
    # Convert the audio data to an AudioFile object
    audio_file = sr.AudioFile(audio_data)
    # Extract the audio data from the file
    with audio_file as source:
        audio = r.record(source)
    # Transcribe the speech using Google Speech Recognition
    text = r.recognize_google(audio)
    return text

def chatbot(input_data):
    # Check if the input data is text or audio
    if isinstance(input_data, str):
        # The input data is text, so pass it directly to the chatbot algorithm
        text = input_data
    else:
        # The input data is audio, so transcribe it to text first
        text = transcribe_speech(input_data)
    # Find the most relevant sentence using the chatbot algorithm
    most_relevant_sentence = get_most_relevant_sentence(text)
    # Return the answer
    return most_relevant_sentence

def main():
    st.title("Speech Recognition Chatbot")
    st.write("Type a message or upload an audio file to start:")

    # add a text input for the user's message
    text_input = st.text_input("Type your message here:")

    # add a file uploader for the user's audio file
    uploaded_file = st.file_uploader("Or choose an audio file:")

    # check if the user provided any input
    if text_input or uploaded_file is not None:
        # pass the user's input to the chatbot function and display the response
        response = chatbot(text_input or uploaded_file)
        st.write("Chatbot: " + response)

if __name__ == "__main__":
    main()
