import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

st.markdown("<h1 style = 'color: #000000; text-align: center; font-family: Courier'>Churn Predictor</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #000000; text-align: center; font-family: Courier '>Built by Adebayo </h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('pngwing.com (2).png')

st.header('Project Info', divider= True)
st.write("The primary objective of this project is to develop a sophisticated predictive model focused on forecasting churn rates within startup subscription services. Leveraging advanced machine learning techniques, the goal is to provide stakeholders with deep insights into the factors influencing subscription cancellations and customer attrition.Through the analysis of extensive datasets, the project aims to empower decision-makers with a comprehensive understanding of the dynamics impacting the success and sustainability of startup ventures. Problem statement is to  develop a predictive model to anticipate churn behavior among subscribers to startup services. Identify key factors contributing to churn within the context of startup subscription models. Provide actionable insights to stakeholders to mitigate churn and enhance customer retention strategies.")

st.markdown("<br>", unsafe_allow_html = True)

ds = pd.read_csv('expresso_processed.csv')
st.dataframe(ds)

sel_cols = ['MONTANT', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'CHURN', 'REGULARITY', 'TENURE', 'MRG']


st.sidebar.image('pngwing.com (6).png', caption = 'Welcome User > _ < ')

st.sidebar.markdown("<br>", unsafe_allow_html = True)
st.sidebar.markdown("<br>", unsafe_allow_html = True)

st.sidebar.subheader('Input your Variables', divider = True)

Mt = st.sidebar.number_input('MONTANT', ds['MONTANT'].min(), ds['MONTANT'].max())
Rev = st.sidebar.number_input('REVENUE', ds['REVENUE'].min(), ds['REVENUE'].max())
Seg = st.sidebar.number_input('ARPU_SEGMENT', ds['ARPU_SEGMENT'].min(), ds['ARPU_SEGMENT'].max())
Freq = st.sidebar.number_input('FREQUENCE', ds['FREQUENCE'].min(), ds['FREQUENCE'].max())
Data = st.sidebar.number_input('DATA_VOLUME', ds['DATA_VOLUME'].min(), ds['DATA_VOLUME'].max())
Net = st.sidebar.number_input('ON_NET', ds['ON_NET'].min(), ds['ON_NET'].max())
Reg = st.sidebar.number_input('REGULARITY', ds['REGULARITY'].min(), ds['REGULARITY'].max())
Ten = st.sidebar.selectbox('TENURE', ds['TENURE'].unique())
mrg = st.sidebar.selectbox('MRG', ds['MRG'].unique())


input_var = pd.DataFrame()
input_var['MONTANT'] = [Mt]
input_var['REVENUE'] = [Rev]
input_var['ARPU_SEGMENT'] = [Seg]
input_var['FREQUENCE'] = [Freq]
input_var['DATA_VOLUME'] = [Data]
input_var['ON_NET'] = [Net]
input_var['REGULARITY'] = [Reg]
input_var['TENURE'] = [Ten]
input_var['MRG'] = [mrg]

st.markdown("<br>", unsafe_allow_html= True)
# display the users input variable 
st.subheader('Users Input Variables', divider= True)
st.dataframe(input_var)


Mt = joblib.load('MONTANT_scaler.pkl')
Rev = joblib.load('REVENUE_scaler.pkl')
Seg = joblib.load('ARPU_SEGMENT_scaler.pkl')
Data = joblib.load('DATA_VOLUME_scaler.pkl')
Ten = joblib.load('TENURE_encoder.pkl')
mrg = joblib.load('MRG_encoder.pkl')



# transform the users input with the imported scalers 
input_var['MONTANT'] = Mt.transform(input_var[['MONTANT']])
input_var['REVENUE'] = Rev.transform(input_var[['REVENUE']])
input_var['ARPU_SEGMENT'] = Seg.transform(input_var[['ARPU_SEGMENT']])
input_var['DATA_VOLUME'] = Data.transform(input_var[['DATA_VOLUME']])
input_var['TENURE'] = Ten.transform(input_var[['TENURE']])
input_var['MRG'] = mrg.transform(input_var[['MRG']])

model = joblib.load('Exxpresso_CHURN.model.pkl')
predicted = model.predict(input_var)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

if st.button('Predict Churn'):
    if predicted == 0:
        st.failure('Customer Has CHURNED')
    else:
        st.success('Customer Is With Us')