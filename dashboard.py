# Bibliothèques
import pandas as pd
import numpy as np
import streamlit as st
import requests
from lime import lime_tabular
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import json

def request_prediction(idClient) : 
    request = requests.post(url="https://annepiercep7.datartichaut.com/prediction",
                        data=json.dumps({'idClient': idClient}),
                        headers={"Content-Type": "application/json"})
    
    return request.json()['resultat']

# Données       
df = pd.read_csv('dataReduced.csv')
model = pickle.load(open('model.pkl', 'rb'))

# Création des onglets
tab1, tab2= st.tabs(["Prêt", "Informations client"])

# -----------------------------------------------------------------------
# Onglet 1 : Score du client via algorithme et API

with tab1:
    st.title('Attribution de prêt')

    idClient = st.number_input('Identifiant client', min_value=0, step=1, value=263934)

    predict_btn = st.button('Prédire')   
    pred = 0
    
    if predict_btn:
        data = idClient
        pred = request_prediction(idClient)
    
    col1, col2, col3 = st.columns([2, 1, 6])

    with col1:
    # jauge
        st.subheader('Score')
        value = pred
        fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = value * 100,
        mode = "gauge+number",
        gauge = {'axis': {'range': [None, 100]},
             'steps' : [
                 {'range': [0, 50], 'color': "red"},
                 {'range': [50, 75], 'color': "orange"},
                 {'range': [75, 100], 'color': "green"}],
             'threshold' : {'line': {'color': "blue", 'width': 4}, 'thickness': 0.75, 'value': value}}))

        st.plotly_chart(fig, use_container_width=True)

    with col3:
    # lime
        st.subheader('Importances')
        # Définir l'explainer
        feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV']]
        X = df[feats].values 
        explainer = lime_tabular.LimeTabularExplainer(X, mode="classification",feature_names=feats)
        # L'appliquer à la ligne d'index de l'idClient
        idx = df.index[df["SK_ID_CURR"] == idClient]
        exp = explainer.explain_instance(X[idx][0], model.predict_proba)
        # graphique
        fig2 = exp.as_pyplot_figure()
        st.pyplot(fig2, use_container_width=True)
        
    with st.container(): 
        # metrics
        st.subheader('Decision')
        if value > 0.70:
            decision = "prêt refusé"
        else :
            decision = "prêt accordé"
        st.metric(label = '.',value = decision)
        
# -----------------------------------------------------------------------
# Onglet 2: visualisation d'info sur le client, d'après dataset en local

with tab2:
    st.header("Infos client")
    yes = df.loc[df['TARGET']==0]
    no = df.loc[df['TARGET']==1]
    client = df.loc[df['SK_ID_CURR']==idClient]

    col1, col2 = st.columns(2)

    with st.container():
        with col1:
            option1 = st.selectbox('1er choix',('AMT_CREDIT', 'AMT_INCOME_TOTAL', 'CC_CNT_INSTALMENT_MATURE_CUM_SUM'))      

        with col2: 
            option2 = st.selectbox('1er choix', ('BURO_CREDIT_DAY_OVERDUE_MEAN', 'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN', 'BURO_AMT_CREDIT_SUM_LIMIT_MEAN'))
            option3 = st.selectbox('2e choix', ('AMT_CREDIT', 'AMT_INCOME_TOTAL'))


    with st.container():
        col3, col4 = st.columns(2)

        with col3:
            st.caption(option1)
            fig3, ax = plt.subplots()
            ax.boxplot([yes[option1], client[option1], no[option1]], showfliers=False, labels=('Yes','client','No'), patch_artist=True)
    
            st.pyplot(fig3)

        with col4:
            st.caption("Votre position : ")
            fig4, bx = plt.subplots()
            plt.xlabel(option2)
            plt.ylabel(option3)
            bx.scatter(yes[option2], yes[option3], color='blue', s=1)
            bx.scatter(no[option2], no[option3], color='green', s=1)
            bx.scatter(client[option2],client[option3], color='red', s=10)
            st.pyplot(fig4)


    with st.container():
        col5, col6 = st.columns(2)

        with col5:
            st.caption('Revenu annuel')
            fig5, ax = plt.subplots()
            ax.hist(df['AMT_INCOME_TOTAL'], bins=20)
            ax.axvline(x=client['AMT_INCOME_TOTAL'].iloc[0], c='orange', linewidth=2)
            st.pyplot(fig5)

        with col6:
            st.caption('Montant du crédit demandé/Revenu annuel')
            fig6, bx = plt.subplots()
            plt.xlabel('Revenu annuel')
            plt.ylabel('Montant du crédit demandé')
            bx.scatter(yes['AMT_INCOME_TOTAL'],yes['AMT_CREDIT'], color='blue', s=1)
            bx.scatter(no['AMT_INCOME_TOTAL'],no['AMT_CREDIT'], color='green', s=1)
            bx.scatter(client['AMT_INCOME_TOTAL'],client['AMT_CREDIT'], color='red', s=5)
            st.pyplot(fig6)
            
