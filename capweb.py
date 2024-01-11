import streamlit as st
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from PIL import Image
import pickle

#images
stock = Image.open("images/programmingStock.png")
codep = Image.open("images/myCode.png")
signp = Image.open("images/sign.png")
megaphone = Image.open("images/Megaphone.png")
puzzle = Image.open("images/advocacyPuzzle.png")
thinking = Image.open("images/thinking.png")
earth = Image.open("images/earth.png")
portrait = Image.open("images/portraitog.png")

with open(r'C:\Users\evanb\OneDrive\Documents\Python_Scripts\models.pkl', 'rb') as model_file:
    loaded_mnom = pickle.load(model_file)

with open(r'C:\Users\evanb\OneDrive\Documents\Python_Scripts\vectorizers.pkl', 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

primarColor = "0000FF"

st.set_page_config(page_title="Spam Detector", page_icon=":e_mail:",)

# about section
with st.container():
    st.title("Python and Machine Learning")
    st.write("___")

    st.header("Why Coding?")
    image_column, text_column = st.columns((1,2))
    with image_column:
        st.image(stock)
    with text_column:
        st.write("I have always had an affinity for computer programming and have plans to pursue it in college, and "
                 "because of that I wanted it to be a part of my capstone project. I decided that I would incorperate "
                 "itby learnning a new language and gaining an understandinng of how machine learning works to then "
                 "create a final project utilizing both")

    st.header("Why Python?")
    st.write("After doing some research into machine learning and programming languages, I found that python is the "
             "best fit and most used language for machine learning code. After further research into python, I found it"
             " to also be one of the most versatile programming languages, as well as one of the most popular "
             "languages, known in some capacity by about half of all programmers.")

    image_column, text_column = st.columns((1,2))
    with image_column:
        st.image(codep)
    with text_column:
        st.header("Final Product")
        st.write("For my final project, using both Python and machine learning, I created a classifier that determines"
                  " whether an email is spam or not by looking at its contents. This proof of concept is to demonstrate"
                  " my new skills in programming, not just to all of you, but to myself aswell.")

# portrait of a graduate
with st.container():
    st.header("Portrait of a Graduate")
    st.image(portrait)

    text_column, image_column = st.columns((2,1))
    with text_column:
        st.header("Globally & Environmentally Aware")
        st.write("Whether it is to predict crop yields, weather patters, or potential outbreaks, Python and machine "
                 "learning are globally applicable and will, as time passes and technology advances have uses in many"
                 " more applications")
    with image_column:
        st.image(earth)

    image_column1, image_column2 = st.columns((1,1))
    with image_column1:
        st.image(signp)
    with image_column2:
        st.image(puzzle)

    text_column1, text_column2 = st.columns((1,1))
    with text_column1:
        st.header("Confidence")
        st.write("When planning my capstone, I had the confidence to be able to teach myself a whole new programming"
                 " language and to build my final project")
    with text_column2:
        st.header("Self Advocacy")
        st.write("Self advocacy represented itself in my project in how I taught myself a whole new way to"
                 " program , and not only learned it, but was able to apply it as well")

    image_column, text_column = st.columns((1,2))
    with image_column:
        st.image(megaphone)
    with text_column:
        st.header("Communication")
        st.write("Throughout both the learning process and the time I spent making my program, I did ask for help from"
                 " my advisor, my teachers, and my peers")

    text_column, image_column = st.columns((2,1))
    with text_column:
        st.header("Problem Solving")
        st.write("If coding is anything, it's problem solving, throughout learning Python I ran into issues along the "
                 "way especially towards the beginning. Problem solving was very prevalent in my time working on the "
                 "final program, every other word I wrote seemed to end up in an error, but regardless I managed to "
                 "finish my project, fixing and rewriting as I went.")
    with image_column:
        st.image(thinking)

# email detector section
with st.container():
    st.title("Spam Email Detector")

    t_in = st.text_input("")


    email_sample = [t_in]
    email_token = loaded_vectorizer.transform(email_sample)

    prediction = loaded_mnom.predict(email_token)

    verdict = st.subheader("")

    if t_in != '':
        if prediction == 0:
            verdict = st.subheader("Not Spam")
        elif prediction == 1:
            verdict = st.subheader("Spam")
    elif t_in == '':
        verdict = st.subheader("")

