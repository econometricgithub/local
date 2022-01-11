import pandas as pd

"""
This is the function to get the regression main result. 
Coeff, Std Err, t, P value , 
"""
def results_summary_to_dataframe(results):
    import pandas as pd
    # take the result of an statsmodel results table and transforms it into a dataframe
    coeff = round(results.params, 3)
    std_err = round(results.bse, 3)
    t = round(results.tvalues, 3)
    p_t = round(results.pvalues, 3)
    conf_lower = round(results.conf_int()[0], 3)
    conf_higher = round(results.conf_int()[1], 3)

    results_df = pd.DataFrame({"coeff": coeff,
                               "std err": std_err,
                               "t": t,
                               "P>| t |": p_t,
                               "conf_lower": conf_lower,
                               "conf_higher": conf_higher
                               })
    # Reordering...
    results_df = results_df[["coeff", "std err", "t", "P>| t |", "conf_lower", "conf_higher"]]
    return results_df


def results_two(results):
    R2 = round(results.rsquared, 3)
    adj_r2 = round(results.rsquared_adj, 3)
    f_value = round(results.fvalue, 3)
    pro_f = round(results.f_pvalue, 3)
    A_iC = round(results.aic, 3)
    B_ic = round(results.bic, 3)
    results_2={"          R-squared         ": R2,
                "     Adj.R-squared       ": adj_r2,
                    "      F-STATISTIC        ": f_value,
                    "      Prob (F-statistic)      ": pro_f,
                    "       AIC       ": A_iC,
                    "       BIC         ": B_ic}
    return results_2

# this function is none - sense . This is only to get the variable name
def result_one(model,y,results):
    import datetime
    import numpy as np
    #Big mistake which read the 1st row of dataset
    #he tried to get the variable name such as Political Efficacy ...
    dv = y[0]
    if dv==4:
        dv= "Political Efficacy"
    if dv==0.55:
        dv = " Online Expressive Political Participation "
    if dv==0.75:
        dv = " Offline Expressive Political Participation "
    new_model = model
    Date = datetime.datetime.now()
    num_observs = np.int(results.nobs)  # number of observations
    results_1 = {"Dependent Variable": dv,
                     "Model": new_model,
                     "Date": Date,
                     "Num of Observs": num_observs}
    return results_1


def main_fun(mm,dv, idv=[]): #mm - model , dv , idv
    import statsmodels.api as sm
    X = sm.add_constant(idv)
    y = dv
    model_m=mm
    X = sm.add_constant(X)
    mlr_results = sm.OLS(y, X).fit()
    dataframe_1 = result_one(model_m,y,mlr_results) #just sring
    dataframe_2 = results_two(mlr_results) #summary result for model
    dataframe_3 = results_summary_to_dataframe(mlr_results) #regression results
    reg_metric_df=reg_metric(y, X,mlr_results) #last table

    return  dataframe_1, dataframe_2, dataframe_3, reg_metric_df

def reg_metric(dv, idv,results):
    import numpy as np
    from bioinfokit.analys import get_data, stat
    res = stat()
    res.reg_metric(y=np.array(dv), yhat=np.array(results.predict(idv)), resid=np.array(results.resid))
    return res.reg_metric_df


#Checking Multiple Linear Regression Assumptions Test

#Linearity and normality Test
def linerity(df_data):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style='ticks', color_codes=True, font_scale=0.5)
    g = sns.pairplot(df_data, height=1, diag_kind='hist', kind='reg')
    g.fig.suptitle('Scatter Plot', y=1.08)
    plt.savefig("static/linearity1.png", dpi=72)
# Homoscedasticity
def homoscedasticity(y, results):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 3.5))
    pred_val = results.fittedvalues.copy()
    true_val = y.values.copy()
    resid = true_val - pred_val
    res = sns.residplot(resid, pred_val)
    plt.title('Homoscedasticity')
    #plt.show()
    plt.savefig("static/homoscedasticity.png", dpi=72)
# normality of errors/residuestatic
def normality(y, results):
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    pred_val = results.fittedvalues.copy()
    true_val = y.values.copy()
    resid = true_val - pred_val
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.probplot(resid, dist='norm', plot=plt)
    #plt.show()
    plt.savefig("static/normality.png", dpi = 72)
#Multicollinearity Test
def multicollinearity(df_data):
    corr=df_data.corr()
    #print(corr)
    return corr

#def slr(model,x,y):
