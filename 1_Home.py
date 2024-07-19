import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components

st.set_page_config(page_title="Home")
st.title("Linear Regression Model")
st.subheader("Performing linear regression with streamlit on AMES Dataset")
st.sidebar.success("Select a function.")

#uploaded_file = st.file_uploader("Upload only csv files")

#if uploaded_file:
# Check if you've already initialized the data
#if 'df' not in st.session_state:
    # Get the data if you haven't
df = pd.read_csv('ames_train.csv')
    # Save the data to session state
st.session_state['df'] = df

# Retrieve the data from session state
df = st.session_state.df
#df = pd.read_csv('ames_train.csv')

    
st.subheader("About the dataset:")
st.write("The Ames Housing Dataset is a well-known dataset in the field of machine learning and data analysis. It contains various features and attributes of residential homes in Ames, Iowa, USA. The dataset is often used for regression tasks, particularly for predicting housing prices.")
# head
if st.checkbox("Preview Dataset"):
    number = st.number_input("Number of rows to view",5)
    st.dataframe(df.head(number))
    
# Show columns
if st.checkbox("Column Names"):
    st.write(df.columns)
    
# Show Shape
if st.checkbox("Shape of Dataset"):
    data_dim = st.radio("Show Dimension By: ", ("Rows","Columns"))
    if data_dim == "Rows":
        st.text("Number of Rows:")
        st.write(df.shape[0])
    elif data_dim == "Columns":
        st.text("Number of Columns: ")
        st.write(df.shape[1])
    else:
        st.write(df.shape)
    
# Show Datatypes
if st.checkbox("Data Types"):
    st.text("Data Types")
    st.write(df.dtypes)

# ----------------------------------------

# -----------------------------------------------
# st.subheader("Data Scaling and Pre-Processing")


