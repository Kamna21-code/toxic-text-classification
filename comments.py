# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 17:30:08 2020

@author: 91782
"""
import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pickle
import nltk
# load Vectorizer For Gender Prediction
#news_vectorizer = open("vector.pkl","rb")
#news_cv = joblib.load(news_vectorizer)

# # load Model For Gender Prediction
# news_nv_model = open("models/naivebayesgendermodel.pkl","rb")
# news_clf = joblib.load(news_nv_model)




# Get the Keys
def get_key(val,my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key



def main():
    """News Classifier"""
    st.title("News Classifier")
    
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h1 style="color:white;text-align:center;">Streamlit ML App </h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    activity = ['Prediction','NLP']
    choice = st.sidebar.selectbox("Select Activity",activity)


    if choice == 'Prediction':
        st.info("Prediction with ML")

        news_text = st.text_area("Enter News Here","Type Here")
        news_text = [news_text]
        all_ml_models = ["LR","SVC","NB"]
        model_choice = st.selectbox("Select Model",all_ml_models)

        prediction_labels = {'toxic': 0,'severe_toxic': 1,'obscene': 2,'threat': 3,'insult': 4,'identity_hate': 5}
        if st.button("Classify"):
            st.text("Original Text::\n{}".format(news_text))

            if model_choice == 'LR':
                pickle_in = open("LR.pkl","rb")
                predictor1= pickle.load(pickle_in)
                prediction = predictor1.predict(news_text)
                # st.write(prediction)
            elif model_choice == 'SVC':
                pickle_in = open("SVC.pkl","rb")
                predictor2 = pickle.load(pickle_in)
                prediction = predictor2.predict(news_text)
                # st.write(prediction)
            elif model_choice == 'NB':
                pickle_in = open("NBC.pkl","rb")
                predictor3 = pickle.load(pickle_in)
                prediction = predictor3.predict(news_text)
                # st.write(prediction)
                # st.write(prediction)

            final_result = get_key(prediction,prediction_labels)
            st.success("News Categorized as:: {}".format(final_result))

    if choice == 'NLP':
        st.info("Natural Language Processing of Text")
        raw_text = st.text_area("Enter News Here","Type Here")
        if st.checkbox("WordCloud"):
            c_text = raw_text
            wordcloud = WordCloud().generate(c_text)
            plt.imshow(wordcloud,interpolation='bilinear')
            plt.axis("off")
            st.pyplot()
    st.sidebar.subheader("About")

if __name__ == '__main__':
    main()
