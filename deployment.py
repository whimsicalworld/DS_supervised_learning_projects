#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np
from sklearn import metrics
import io

st.title("Regression Model Deployment")
st.sidebar.title("Regression Parameters")

st.header("Load and Preprocess Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Loaded Successfully!")
    st.write(data.head())
    option = st.sidebar.selectbox('Would you like to get the Descriptive Statistics of the Data?',('Yes', 'No'))
    if option == 'Yes':
     st.write(data.describe())
    else:
     st.header("Types of data types in Data")
    # Display data info
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    option1 = st.sidebar.selectbox('Would you like to get the Histogram plots of the Data?',('Yes', 'No'))
    if option == 'Yes':
     #displaying histogram
     for column in data:
      st.write(f'#### {column}')
      fig, ax = plt.subplots()
      sns.histplot(data[column], ax=ax)
      st.pyplot(fig)
    else:
     st.write('ok')
    # Plot heatmap
    st.write('### Heatmap of Correlation Matrix')
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    data.columns = ['solar_noon', 'temp', 'wind_dir', 'wind_spd', 'sky_cov', 'vis', 'hum', 'avg_wind_spd', 'avg_pres', 'power']
    data['avg_wind_spd'].fillna(data['avg_wind_spd'].mean(), inplace=True)
    option2 = st.sidebar.selectbox('Would you like to build the model?',('Yes', 'No'))
    if option2 =='Yes':
        import statsmodels.formula.api as smf 
        model=smf.ols('power~solar_noon+temp+wind_dir+wind_spd+sky_cov+vis+hum+avg_wind_spd+avg_pres',data=data).fit()
        st.write(model.summary())
    else:
        st.write('You cannot proceed further')
    option3 = st.sidebar.radio("Would you like to get the Inferences from above summary",('Yes', 'No'))

    if option3 == 'Yes':
      st.write('The Prob of F-statistic indicates high significance of coefficients as it is completely zero')
      st.write('0.64 means that 64% of the variance in the dependent variable can be explained by the independent variables in the model.')
      st.write('So if the regression coefficients have to be significant the observed p values shall be less than 0.05')
      st.write('A lower AIC, BIC values indicates a better model.')
      st.write('Since the probabilities of omnibus , JB is less than 0.05 ,Hence the residuals are not normally distributed')
      st.write('From durbin - Watson values, We can confirm that observations are a bit independent as it is around 2')
    else:
     st.write("Hope you'll understand.")
    
    
#     fig, ax = plt.subplots()
#     sm.qqplot(residuals, line='45', ax=ax)
#     st.write('### Normal Q-Q plot of residuals')
#     st.pyplot(fig)
    
    #Data preprocessing
    X = data.drop('power', axis = 1)
    y = data['power']    
    
    # Splitting the dataset
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3)
    random_state = st.sidebar.slider("Random state", 0, 100, 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import BaggingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.neighbors import KNeighborsRegressor
    import xgboost as xgb
    models = []

    names = [
    "LinearRegression",
    "DecisionTreeRegressor",
    "BaggingRegressor",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
    "AdaBoostRegressor",
    "KNeighborsRegressor",
    "xgb.XGBRegressor"
     ]

    scores = []

    clf = [
    LinearRegression(),
    DecisionTreeRegressor(),
    BaggingRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    AdaBoostRegressor(),
    KNeighborsRegressor(),
    ]
    for model in clf:
     model.fit(X_train, y_train)
     score = model.score(X_test, y_test)
     scores.append(score)
    # Display results
    st.header("Model Results")
    final_scores = pd.DataFrame(zip(names,scores), columns=['Classifier', 'Accuracy'])
    st.write(final_scores)
    final_scores.sort_values(by='Accuracy',ascending=False).style.background_gradient(cmap="tab10").set_properties(**{
            'font-family': 'Comic Sans MS',
            'color': 'Brown',
            'font-size': '15px'
        })
    
    def print_evaluate(true, predicted, train=True):  
     mae = metrics.mean_absolute_error(true, predicted)
     mse = metrics.mean_squared_error(true, predicted)
     rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
     r2_square = metrics.r2_score(true, predicted)
     if train:
        st.write("========Training Result=======")
        st.write('MAE: ', mae)
        st.write('MSE: ', mse)
        st.write('RMSE: ', rmse)
        st.write('R2 Square: ', r2_square)
     elif not train:
        st.write("=========Testing Result=======")
        st.write('MAE: ', mae)
        st.write('MSE: ', mse)
        st.write('RMSE: ', rmse)
        st.write('R2 Square: ', r2_square)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    gbr = GradientBoostingRegressor(learning_rate=0.03,max_depth=5,random_state=42)
    gbr.fit(X_train,y_train)
    prediction = gbr.predict(X_test)
    st.write('Prediction Values')
    st.write((prediction))
    print_evaluate(y_train, y_train_pred, train=True)
    print_evaluate(y_test, y_test_pred, train=False)
    cross_checking = pd.DataFrame({'Actual' : y_test , 'Predicted' : prediction})
    st.write(cross_checking.head())
    cross_checking['Error'] = cross_checking['Actual'] - cross_checking['Predicted']
    st.write(cross_checking.head())
    cross_checking_final  = cross_checking[cross_checking['Error'] <= 20]
    cross_checking_final.sample(25).style.background_gradient(
        cmap='Dark2').set_properties(**{
            'font-family': 'Times New Roman',
            'color': 'LigntGreen',
            'font-size': '15px'
        })
    






























































