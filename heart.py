import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
st.header('Heart Diseases Prediction Created By Cole')

dg = pd.read_csv("heart-disease.csv")
st.dataframe(dg)

df = dg[['age',
         'sex',
         'cp',
         'trestbps',
         'chol',
         'fbs',
         'target']].dropna()


x = df[['age',
        'sex',
        'cp',
        'trestbps',
        'chol',
        'fbs',]]
y = df[['target']]

#20% of the dataset is for testing and 70% of the dataset is for training
feature_train, feature_test, target_train, target_test = train_test_split(x, y, test_size=0.2)

model = LogisticRegression()
model.fit(feature_train, target_train)

#student portal features which will be written on the sidebar
st.sidebar.title('Heart Disease features')

age =st.sidebar.slider('age', min_value=0)
sex = st.sidebar.selectbox('Gender', ['Male', 'Female'])
cp = st.sidebar.number_input('Chestpain', min_value=0)
trestbps = st.sidebar.number_input('Resting_Blood_Pressure', min_value=0)
chol = st.sidebar.number_input('Cholesterol', min_value=0)
fbs = st.sidebar.slider('Fasting_Blood_Sugar', min_value=0)



if sex == 'Male':
    sex = 1
else:
    sex = 0

total = {
         'age': [age],
         'sex': [sex],
         'cp': [cp],
         'trestbps': [trestbps],
         'chol': [chol],
         'fbs': [fbs]}

#print(total)
st.write('Heart Details')
st.dataframe(total, width=700)
pf = pd.DataFrame(total)
st.write('1 means the patient is prone to heart disease')
st.write('0 means the patient is safe from heart disease')

if st.button('Check Prediction'):
    prediction = model.predict(pf)
    #st.write('The possibility of heart disease is', prediction)
    st.write(f'The possibility of heart disease is: {prediction[0]}')


