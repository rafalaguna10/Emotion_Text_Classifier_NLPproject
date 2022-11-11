from cProfile import label
from secrets import choice
from turtle import color
from unittest import result
import streamlit as st
import altair as alt

import pandas as pd
import numpy as np

import joblib


pipe_lr = joblib.load(open("model/emotion_text_classifier_model.pkl","rb"))

def predict_emotions(docx):
    resuls = pipe_lr.predict([docx])
    return resuls[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


 
def main():
    st.title("Emotion Classifier App")
    menu = ["Home","Monitor","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home-Emotion In Text")

        with st.form(key='emotion_form'):
            raw_text = st.text_area("Type a sentence please")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1,col2 = st.columns(2)

            #Apply funcs
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)


            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Emotion text prediction")
                st.write(prediction)
                st.write("COnfidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                #st.write(probability)
                proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
                #st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions","probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability', color='emotions')
                st.altair_chart(fig,use_container_width=True)

    elif choice == "Monitor":
        st.subheader("Monitor App")

    else:
        st.subheader("About")

if __name__ == '__main__':
    main()