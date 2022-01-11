#importing necessary library
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template,flash, request
from wtforms import Form, StringField, validators, SubmitField, SelectField
from bioinfokit.analys import get_data, stat
#Importing SQLAlcheny
from flask_sqlalchemy import SQLAlchemy
#from custom_validators import height_validator, weight_validator
from myModules import results_summary_to_dataframe, result_one,results_two, main_fun, reg_metric, linerity,normality,homoscedasticity,multicollinearity
from task import model_1
app = Flask(__name__)
#result_one, result_two,result_three, reg_metric_output=main_fun(y,X)
#linerity(df2)
#normality(y, regresion_results)
#homoscedasticity(y, regresion_results)
#corr=multicollinearity(df2)
@app.route('/')
def home():
    return render_template('main.html')
@app.route('/background')
def background():
    return render_template("background.html")

#this function is for Stat DataAnalysis MLR ***
@app.route('/sda', methods = ['GET', 'POST'])
def sda():
    #upload the CSV to Data Frame
    df = pd.read_csv("Dataset/data_cleaning.csv")
    idv=[] #Indenpendent Variable - Format array
    dv=" " #single string
    model="" #chosen model
    if request.method=='POST':
        idv=request.form.getlist("idv")
        dv=request.form.get("dv")
        model=request.form.get("model")
        X=df[idv]
        y=df[dv]
        #X=sm.add_constant(X)
        #regresion_results=sm.OLS(y,X).fit()
        #result_one -
        result_one, result_two, result_three, reg_metric_output = main_fun(model,y, X)
        return render_template('output.html',table= result_one, tables1=result_two,  tables2=[result_three.to_html(classes='output2')], tables3=[reg_metric_output.to_html(classes='output3')])
    return render_template("sda.html")
@app.route('/slr', methods = ['GET', 'POST'])
def slr():
    df = pd.read_csv("Dataset/data_cleaning.csv")
    idv =" "
    dv = " "
    if request.method == 'POST':
        idv = request.form.get("IDV")
        dv = request.form.get("DV")

        a,b,c,d=model_1(OFEP=dv,SIU=idv)
        return render_template("slroutput.html",tables=a, tables1=b,tables2=[c.to_html(classes='output2')],tables3=[d.to_html(classes='output3')], independent=idv)
    return render_template("slr.html")
@app.route('/descriptive', methods = ['GET', 'POST'])
def descriptive():
    return render_template("descriptive.html")
if __name__ == '__main__':
    app.run(debug=True)

