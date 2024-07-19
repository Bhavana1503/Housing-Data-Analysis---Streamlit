import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
#st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Data Modelling")

df_clean = st.session_state['df_clean']

y = df_clean["SalePrice"]
X = df_clean.drop('SalePrice', axis=1)

st.write(df_clean.dtypes.value_counts())

if st.checkbox("Check datatypes of dependent variables"):
    st.write(X.dtypes)

if st.checkbox("Check datatype of independent variables"):
    st.write(y.dtypes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

st.write("Now we have split the dataset into training and testing sets.")

if st.checkbox("Check the shape of training set"):
    st.write("Dependent Variables",X_train.shape)
    st.write("Independent Variables",y_train.shape)

if st.checkbox("Check the shape of testing set"):
    st.write("Dependent Variables",X_test.shape)
    st.write("Independent Variables",y_test.shape)

st.subheader("OLS Model")
X_train = sm.add_constant(X_train)
y_train.values.reshape(-1,1)

ols_model = sm.OLS(y_train.astype(float), X_train.astype(float)).fit()

if st.checkbox("See OLS Model Summary"):
    st.write(ols_model.summary())

st.write("We now drop insignificant variables based on p-value threshold (e.g., 0.05).")

# Drop insignificant variables based on p-value threshold (e.g., 0.05)
significant_vars = ols_model.pvalues[ols_model.pvalues < 0.05].index
X_significant = X_train[significant_vars]
if st.checkbox("See significant variables"):
    st.write(X_significant)

st.subheader("OLS model with significant variables")

st.write("We now fit OLS model with significant variables only")
ols_model_significant = sm.OLS(y_train.astype(float), X_significant.astype(float))
result=ols_model_significant.fit()

if st.checkbox("See Significant Variables OLS Model Summary"):
    st.write(result.summary())

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predict y_train using the trained model
y_train_pred = ols_model.predict(X_train)

# Calculate Mean Squared Error (MSE) on training data
mse_train = mean_squared_error(y_train, y_train_pred)

# Calculate Root Mean Squared Error (RMSE) on training data
rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)

# Calculate Mean Absolute Error (MAE) on training data
mae_train = mean_absolute_error(y_train, y_train_pred)

# Calculate R-squared (R^2) on training data
r2_train = r2_score(y_train, y_train_pred)

if st.checkbox("Evaluation metrics on training data"):
    st.write("Training Mean Squared Error (MSE):", mse_train)
    st.write("Training Root Mean Squared Error (RMSE):", rmse_train)
    st.write("Training Mean Absolute Error (MAE):", mae_train)
    st.write("Training R-squared (R^2):", r2_train)

# ------------------------------------------

# Predict y_train using the trained model
#y_test_pred = ols_model.predict(X_test)

# Calculate Mean Squared Error (MSE) on training data
#mse_test = mean_squared_error(y_test, y_test_pred)

# Calculate Root Mean Squared Error (RMSE) on training data
#rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)

# Calculate Mean Absolute Error (MAE) on training data
#mae_test = mean_absolute_error(y_test, y_test_pred)

# Calculate R-squared (R^2) on training data
#r2_train = r2_score(y_test, y_test_pred)


st.subheader("Residual Analysis")
residuals = result.resid

# Plotting residuals vs target variable
plt.scatter(y_train, residuals)
plt.xlabel("Target Variable")
plt.ylabel("Residuals")
plt.title("Residuals vs Target Variable")

st.pyplot()

st.subheader("Gradient Based Regressor")
from sklearn.linear_model import SGDRegressor

# Create an instance of SGDRegressor
sgd_model = SGDRegressor()

# Fit the model on the training data
sgd_model.fit(X_train, y_train)

# Predict y_test using the trained model
y_train_pred_sgd = sgd_model.predict(X_train)

# Calculate evaluation metrics
mse_sgd = mean_squared_error(y_train, y_train_pred_sgd)
mae_sgd = mean_absolute_error(y_train, y_train_pred_sgd)
r2_sgd = r2_score(y_train, y_train_pred_sgd)

# Print the evaluation metrics
st.write("Mean Squared Error (MSE):", mse_sgd)
st.write("Mean Absolute Error (MAE):", mae_sgd)
st.write("R-squared (R^2):", r2_sgd)
st.write("-------------------------------")
st.write("Comparing SGD MSE with OLS MSE")
st.write("Mean Squared Error (MSE) of SGD:", mse_sgd)
st.write("Training Mean Squared Error (MSE) of OLS:", mse_train)

st.subheader("Lasso regression")
from sklearn.linear_model import LassoCV

# Fit Lasso regression
lasso_model = LassoCV(cv=5)
lasso_model.fit(X_train, y_train)

# Get selected features
lasso_selected_features = X_train.columns[lasso_model.coef_ != 0]

# Compare selected features with OLS significant variables
st.write("Features selected by Lasso:", lasso_selected_features)
st.write("Features selected by OLS:", X_significant.columns)

st.subheader("Comparing the features dropped by lasso regression to features dropped via insignificant variables, Business understanding.")

st.write("Number of features obtained from lasso regression:", len(lasso_selected_features))
st.write("Number of features obtained from ols method:", X_significant.shape[1])

count = 0
common = []
for i in lasso_selected_features:
    for j in X_significant.columns:
        if i == j:
            common.append(i)
            count = count+1
st.write("Total no. of common variables:", count)
st.write("Common columns observed are:", common)