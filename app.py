#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# In[4]:


def main():

    df = load_data("c://Users/AB/Desktop/10aca/week-3/data/train_store.csv")
    page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Exploration', 'Prediction'])

    if page == 'Homepage':
        st.title('Rossman Sales Prediction')
        st.text('Select a page in the sidebar')
        st.dataframe(df)
    elif page == 'Exploration':
        st.title('Exploration of Rossman Sales Dataset')
        if st.checkbox('Show column descriptions'):
            st.dataframe(df.describe())
        
        st.markdown('### Analysing column relations')
        st.text('Correlations:')
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(df.corr(), annot=True, ax=ax)
        st.pyplot(fig)
        st.text('Effect of the different classes')
        sns.pairplot(df, vars=['CompetitionDistance', 'Customers', 'Open', 'CompetitionDistance'], hue='Sales')
        st.pyplot()
    else:
        st.title('Modelling')
        model, accuracy = train_model(df)
        st.write('Accuracy: ' + str(accuracy))
        st.markdown('### Make prediction')
        st.dataframe(df)
        row_number = st.number_input('Select row', min_value=0, max_value=len(df)-1, value=0)
        st.markdown('#### Predicted')
        st.text(model.predict(df.drop(['alcohol'], axis=1).loc[row_number].values.reshape(1, -1))[0])


# In[5]:


@st.cache
def train_model(df):
    features = ["Store",'DayOfWeek','Promo',
       'StateHoliday', 'SchoolHoliday', 'Year', 'Month', 'DayOfMonth',
       'StoreType', 'Assortment',
       'CompetitionDistance', 'Promo2',
       'CompetitionOpen', 'PromoOpen']
    X = df[features]
    y= df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    return model, model.score(X_test, y_test)


# In[6]:


@st.cache
def load_data(path):
    return pd.read_csv(path)


# In[7]:


if __name__ == '__main__':
    main()


# In[ ]:




