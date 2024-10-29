import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('trained_model.sav','rb'))

st.title("Iris Prediction")

sl = st.number_input("Sepal Length: ")
sw = st.number_input("Sepal Width: ")
pl = st.number_input("Petal Length: ")
pw = st.number_input("Petal Width: ")

input_data = [sl,sw,pl,pw]

result = ""
if st.button("Result"):
    input_data = np.asarray(input_data).reshape(1,-1)
    pred = model.predict(input_data)
    result = pred[0]

st.success(result)