import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

def y_df_RMSE_r2(y_train, y_validate):
    """ Calculare RMSE and r^2 score using a dataframe containing 
        predictions of multiple models """
    # Initialize dataframe to return
    running_df = pd.DataFrame(columns=['Model','Train_RMSE','Validate_RMSE','Train_r2','Validate_r2'])
    for model in y_train.columns[1:]:
        # Calculate values
        rmse_train = mean_squared_error(y_train.actuals, y_train[model]) ** 0.5
        rmse_validate = mean_squared_error(y_validate.actuals, y_validate[model]) ** 0.5
        r2_train = r2_score(y_train.actuals, y_train[model])
        r2_validate = r2_score(y_validate.actuals, y_validate[model])
        # Append values to dataframe
        running_df = running_df.append({'Model':model, 
                                       'Train_RMSE': rmse_train, 'Validate_RMSE': rmse_validate,
                                       'Train_r2': r2_train, 'Validate_r2': r2_validate}, 
                                        ignore_index=True)
    return running_df

def regression_shotgun(X_train, y_train, X_validate, y_validate):
    """ Create several OLS, LASSO+LARS, GLM, and Polynomial regression models,
        Push model predictions to originating dataframe, return dataframe """
    # Baseline
    y_train, y_validate = regression_bl(y_train, y_validate)
    # OLS models
    y_train, y_validate = ols_predictor(X_train, y_train, X_validate, y_validate)
    # LASSO+LARS models
    y_train, y_validate = lars_predictor(X_train, y_train, X_validate, y_validate)
    # GLM models
    # y_train, y_validate = glm_predictor(X_train, y_train, X_validate, y_validate)
    # Polynomial regressions
    y_train, y_validate = pf_lm_predictor(X_train, y_train, X_validate, y_validate)
    
    return y_train, y_validate

def regression_bl(y_train, y_validate):
    """ Create mean and median baseline models, add predictions to y dataframes """
    # Means
    y_train['mean_bl'] = y_train.actuals.mean()
    y_validate['mean_bl'] = y_validate.actuals.mean()
    # Medians
    y_train['median_bl'] = y_train.actuals.median()
    y_validate['median_bl'] = y_validate.actuals.median()
    
    return y_train, y_validate
    
def ols_predictor(X_train, y_train, X_validate, y_validate):
    """ Create OLS model, add predictions to y dataframes """
    # Build, fit model
    lm = LinearRegression(normalize=True).fit(X_train, y_train.actuals)
    # Make predictions
    y_train['ols_preds'] = lm.predict(X_train)
    y_validate['ols_preds'] = lm.predict(X_validate)
    
    return y_train, y_validate

def lars_predictor(X_train, y_train, X_validate, y_validate):
    """ Create LASSO+LARS models, add predictions to y dataframes """
    # Set hyperparameters
    alpha_list = [.0001, .001, .01, .1, 1, 10, 100, 1000]
    # Iterate through each hyperparameter
    for alpha in alpha_list:
        name = 'lars_' + str(alpha) # Generate model name
        # Build, fit, and predict all in one go
        y_train[name + '_preds'] = LassoLars(alpha=alpha).fit(X_train, y_train.actuals).predict(X_train)
        y_validate[name + '_preds'] = LassoLars(alpha=alpha).fit(X_train, y_train.actuals).predict(X_validate)
        
    return y_train, y_validate

def glm_predictor(X_train, y_train, X_validate, y_validate):
    """ Create GLM models, add predictions to y dataframes """
    # Set hyperparameters
    alpha_list = [.0001, .001, .01, .1, 1, 10, 100, 1000]
    power_list = [0,1,2,3]
    # Iterate through each hyperparameter combination
    for power in power_list:
        for alpha in alpha_list:
            name = 'glm_' + 'p' + str(power) + 'a' + str(alpha) # Generate model name
            # Build, fit, and predict all in one go
            y_train[name + '_preds'] = TweedieRegressor(power=power, alpha=alpha).fit(X_train,y_train.actuals).predict(X_train)
            y_validate[name + '_preds'] = TweedieRegressor(power=power, alpha=alpha).fit(X_train,y_train.actuals).predict(X_validate)

    return y_train, y_validate 
            
def pf_lm_predictor(X_train, y_train, X_validate, y_validate):
    """ Create Polynomial Regression models, add predictions to y dataframes """
    # Set hyperparameters
    degree_list = [2,3,4,5,6]
    for degree in degree_list:
        name = 'lm_pf_' + str(degree) # Generate model name
        lm = LinearRegression(normalize=True) # Create linear regression model
        # Create polynomial variables
        X_train_pf = PolynomialFeatures(degree=degree).fit_transform(X_train)
        X_validate_pf = PolynomialFeatures(degree=degree).fit(X_train).transform(X_validate)
        # Make predictions on polynomial variables
        y_train[name + '_preds'] = lm.fit(X_train_pf, y_train.actuals).predict(X_train_pf)
        y_validate[name + '_preds'] = lm.fit(X_train_pf, y_train.actuals).predict(X_validate_pf)
        
    return y_train, y_validate

def plot_residuals(x, y_train):
    """ Creates a residual plot from one variable """
    # Pulls in model names
    model_names = list(y_train.columns[1:])

    # Calculate, plot residuals
    for model_name in model_names:
        y_train[model_name + '_residual'] = y_train['actuals'] - y_train[model_name]
        sns.set(rc={'figure.figsize':(12,8)})
        sns.relplot(x, y_train[model_name + '_residual'], kind='scatter')
        plt.title('Residual Plot for Model: ' + model_name[:-6])
        plt.ylabel('Residuals')
        plt.show()

def regression_errors(y, yhat):
    """ Returns SSE, ESS, TSS, MSE, and RMSE from two arrays """

    # Create dataframe of input values
    df = pd.DataFrame({'y':y, 'yhat':yhat})
    # Calculate errors
    sse = round(mean_squared_error(df.y, df.yhat) * len(y), 2)
    ess = round(sum((df.yhat - df.y.mean())**2), 2)
    tss = round(ess + sse, 2)
    mse = round(mean_squared_error(df.y, df.yhat), 2)
    rmse = round(mse ** 0.5, 2)
    # Organize error calculations in dict
    return_dict = {
        'SSE':sse,
        'ESS':ess,
        'TSS':tss,
        'MSE':mse,
        'RMSE':rmse
    }

    # Return error calculations
    return return_dict

def baseline_mean_errors(y):
    """ Returns baseline model's SSE, MSE, and RMSE from array """

    # Create dataframe
    df = pd.DataFrame({'y':y, 'baseline':y.mean()})
    # Calculate errors
    sse_baseline = round(mean_squared_error(df.y, df.baseline) * len(df), 2)
    mse_baseline = round(mean_squared_error(df.y, df.baseline), 2)
    rmse_baseline = round(mse_baseline ** 0.5, 2)
    # Assign to dict
    return_dict = {
        'Baseline_SSE':sse_baseline,
        'Baseline_MSE':mse_baseline,
        'Baseline_RMSE':rmse_baseline
    }

def better_than_baseline(y, yhat):
    """ Returns True if your regression model performs better than the mean baseline"""

    # Run calculations
    model_errors = regression_errors(y, yhat)
    baseline_errors = baseline_mean_errors(y)
    # Compare model and baseline errors
    better = model_errors['RMSE'] < baseline_errors['Baseline_RMSE']

    # Return True or False for model performing better than baseline
    return better