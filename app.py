from operator import index
import streamlit as st
import plotly.express as px
from pycaret.classification import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 




with st.sidebar:
    st.image('https://etimg.etb2bimg.com/photo/90130074.cms')
    st.title('AutoML Web Application')
    choice = st.radio('Navigation', ["Upload", "Profiling", "Modelling", "Download Model"])
    st.info("Application allows you to build an automated ML pipeline using Streamlit Library")


if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload Your Dataset for Modelling")
    file = st.file_uploader("Upload Dataset here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.info("Press Button Below to make Automated Exploratory Analysis")
    if st.button("Run Profiling"):
        st.title("Autmated Exploratory Data analysis")
        profile_report = df.profile_report()
        st_profile_report(profile_report)



if choice == "Modelling":
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target, silent=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')




if choice == "Download Model": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")