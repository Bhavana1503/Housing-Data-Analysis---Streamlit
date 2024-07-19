import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np

st.title("Exploratory Data Analysis")

df = st.session_state['df']

# EDA
st.subheader("Exploratory Data Analysis")
if st.checkbox("Numerical Columns"):
    st.text("Numerical Columns in the Dataset are: ")
    st.write(df.select_dtypes(exclude = 'object'))
if st.checkbox("Categorical Columns"):
    st.text("Categorical Columns in the Dataset are: ")
    st.write(df.select_dtypes(include = 'object'))

NumCol = df.select_dtypes(exclude = 'object').columns

st.subheader("Correlation Analysis for Numeric Variables")
options = st.multiselect(
    'Select variables for checking correlation',
    NumCol
    )
st.write(options)

if options: 
    # Create a Seaborn correlation plot
    plt.figure(figsize=(20,20))
    plot = sns.heatmap(df[options].corr(), annot=True)
 
    # Display the plot in Streamlit
    st.pyplot(plot.get_figure())

st.subheader("Data Understanding")
st.write("Based on the business understanding, the following columns appear to be irrelavant:")
st.markdown("- MSSubClass: Identifies the type of dwelling involved in the sale.")
st.markdown("- Street: Type of road access to the property.")
st.markdown("- Alley: Type of alley access to the property.")
st.markdown("- LandSlope: Slope of the property.")
st.markdown("- BldgType: Type of dwelling.")
st.markdown("- MasVnrType: Masonry veneer type.")
st.markdown("- MasVnrArea: Masonry veneer area in square feet.")
st.markdown("- FireplaceQu: Fireplace quality.")
st.markdown("- GarageYrBlt: Year garage was built.")
st.markdown("- GarageFinish: Interior finish of the garage.")
st.markdown("- GarageQual: Garage quality.")
st.markdown("- PoolQC: Pool quality.")
st.markdown("- Fence: Fence quality")
st.markdown("- MiscFeature: Miscellaneous feature not covered in other categories")
st.markdown("- MoSold: Month Sold.")

st.write("We now check the correlation of numeric variables from these variables with our target column.")
cor = df.select_dtypes(exclude = 'object').corr()['SalePrice']
st.write(cor.sort_values(ascending = False))

st.write("We drop these irrelavand columns")

df = df.drop(['Id','MSSubClass', 'Street', 'Alley', 'LandSlope','BldgType','MasVnrType',
                    'FireplaceQu','GarageYrBlt','GarageFinish','GarageQual','PoolQC','MiscFeature','Fence','MoSold'], axis=1)

st.write("Now we have: ")
st.text("Number of Rows:")
st.write(df.shape[0])
st.text("Number of Columns:")
st.write(df.shape[1])

plt.figure(figsize=(8,6))
displot = sns.distplot(df['SalePrice'])
title = plt.title("House Price Distribution")

st.pyplot(displot.get_figure())

st.write("We can see that, the target column approximately follows normal distribution")


colnames = list(df.columns)

st.write("Total missing values in the entire training dataset: ", df.isna().sum().sum())

st.write("Columns having missing values in training dataset are: ")
for i in colnames:
    if df[i].isna().sum() != 0:
        st.write(i, df[i].isna().sum())

st.write("'LotFrontage' variable is of float data type. Hence, we replace the missing values with median instead of dropping the columns.")
median_LF = df['LotFrontage'].median()
df['LotFrontage'].fillna(median_LF, inplace = True)

st.write("From the data descrition, the following variables: 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageCond' contains a category 'NA' which is not a missing value")
variables = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageCond']

valcount = st.selectbox(
    'Select the variable for which you want to check the value counts',
    variables)
st.write('You selected:', valcount)
st.write(df[valcount].value_counts())
st.write('---------------------------------')

st.write("But we can see that, it is considered as a missing value. Hence, we fill it with 'NA' variable, considering it as one of the categories.")

for i in variables:
    df[i].fillna("NA", inplace = True)

st.write("Now you can see that a new category 'NA' is created.")

st.write(df[valcount].value_counts())
st.write('---------------------------------')

st.write("In 'Electricity' variable, only 1 row has missing value and hence we can eliminate that row")
df = df.dropna(axis=0)

st.write("After treating the missing values in training dataset, we have:")
st.write("Number of rows: ",df.shape[0])
st.write("Number of columns: ",df.shape[1])

st.write("Total missing values in the entire training dataset: ", df.isna().sum().sum())

st.write(df.head())
CatVar = df.select_dtypes(include=['object'])
NumVar=df.select_dtypes(include=[np.number])

if st.checkbox("Numeric Columns of the dataset"):
    st.write(NumVar.head())

if st.checkbox("Categorical Columns of the dataset"):
    st.write(CatVar.head())

# Select numeric features for scaling
numeric_features = NumVar.columns.difference(["PoolArea"])
minmax_feature = NumVar["PoolArea"]

# Standardize numeric features
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

scaler1 = MinMaxScaler()
df['PoolArea'] = scaler1.fit_transform(df[['PoolArea']])

st.subheader("Cardinality of Categorical columns")
st.write("Cardinality is the number of categories in each categorical column.")
categorical_features = CatVar.columns
cardinality = df[categorical_features].nunique()

if st.checkbox("Check Cardinality"):
    st.write("Cardinality of Categorical Variables:")
    st.table(cardinality)

#high_cardinality_threshold = 10  # Threshold to consider a variable as high cardinality
#high_cardinality_variables = cardinality[cardinality > high_cardinality_threshold].index
#low_cardinality_variables = cardinality[cardinality < high_cardinality_threshold].index
df_clean = pd.get_dummies(df, columns=categorical_features, drop_first=True, dummy_na=True)
#for col in CatVar:
    #if col in high_cardinality_variables:
        #df.drop(columns=[col], inplace=True)
    #else:
        # Use One-Hot Encoding for low cardinality variables
        #df_clean = pd.get_dummies(df, columns=[col], drop_first=True)

st.write("Now we have cleaned dataset ready for model fitting: ", df_clean.head())

st.session_state['df_clean'] = df_clean