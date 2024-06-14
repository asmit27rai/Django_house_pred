from django.shortcuts import render 
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def input(request):
    return render(request, 'input.html')

def result(request):
    data = pd.read_csv("HousingData.csv")
    for col in data.columns:
        data[col].fillna(value=data[col].mean(), inplace=True)
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    d1 = float(request.GET['v1'])
    d2 = float(request.GET['v2'])
    d3 = float(request.GET['v3'])
    d4 = float(request.GET['v4'])
    d5 = float(request.GET['v5'])
    d6 = float(request.GET['v6'])
    d7 = float(request.GET['v7'])
    d8 = float(request.GET['v8'])
    d9 = float(request.GET['v9'])
    d10 = float(request.GET['v10'])
    d11 = float(request.GET['v11'])
    d12 = float(request.GET['v12'])
    d13 = float(request.GET['v13'])

    input_data = np.array([[d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13]])
    
    input_data_scaled = scaler_X.transform(input_data)
    
    pred_scaled = model.predict(input_data_scaled)
    
    pred = scaler_y.inverse_transform(pred_scaled)
    pred = pred[0][0]

    return render(request, "input.html", {"result": round(pred, 2)})
