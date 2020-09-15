import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import statsmodels.api as sm
from dash.dependencies import Input, Output
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt

import dataCreation as dc
from dataCreation import add_result
from supervised import supervised_models

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
application = app.server
app.title = 'Fatemeh'

start_date = "2010-01-02"
end_date = "2020-07-26"

# used for supervised learning only
shift_val = -4
test_period = 500

df = dc.make_initial_data(start_date, end_date)
values = dc.feature_generation(df, start_date, end_date, shift_val)

# Dataframe to keep our results
summary = pd.DataFrame(columns=['Model', 'MAE', 'Prediction', 'Validation'])

sm = supervised_models(values, forecast_period=-shift_val, test_period=test_period)

model_names = ["Random Forest Regressor", "Decision Tree Regressor",
               "Gradient Boosting Regressor",
               "AdaBoost Regressor", "Extra Trees Regressor", "KNeighbors Regressor"]

model_name = "Random Forest Regressor"
filename = 'models/' + model_name + '.sav'
grid_RF = pickle.load(open(filename, 'rb'))
predicted = grid_RF.predict(sm.X_test)
mae_RF = mean_absolute_error(sm.y_test, predicted)
# grid_RF, mae_RF = sm.RandomForestRegressor_model()
forecast_RF = grid_RF.predict(sm.X_forecast)
res = ["Random Forest Regressor", mae_RF, forecast_RF[-1], 'Time Series Cross Validation']
summary = add_result(summary, res)

model_name = "Decision Tree Regressor"
filename = 'models/' + model_name + '.sav'
grid_DT = pickle.load(open(filename, 'rb'))
predicted = grid_DT.predict(sm.X_test)
mae_DT = mean_absolute_error(sm.y_test, predicted)
forecast_DT = grid_DT.predict(sm.X_forecast)
res = ["Decision Tree Regressor", mae_DT, forecast_DT[-1], 'Time Series Cross Validation']
summary = add_result(summary, res)

model_name = "AdaBoost Regressor"
filename = 'models/' + model_name + '.sav'
grid_AB = pickle.load(open(filename, 'rb'))
predicted = grid_AB.predict(sm.X_test)
mae_AB = mean_absolute_error(sm.y_test, predicted)
forecast_AB = grid_AB.predict(sm.X_forecast)
res = ["AdaBoost Regressor", mae_AB, forecast_AB[-1], 'Time Series Cross Validation']
summary = add_result(summary, res)

model_name = "Gradient Boosting Regressor"
filename = 'models/' + model_name + '.sav'
grid_GB = pickle.load(open(filename, 'rb'))
predicted = grid_GB.predict(sm.X_test)
mae_GB = mean_absolute_error(sm.y_test, predicted)
forecast_GB = grid_GB.predict(sm.X_forecast)
res = ["Gradient Boosting Regressor", mae_GB, forecast_GB[-1], 'Time Series Cross Validation']
summary = add_result(summary, res)

model_name = "KNeighbors Regressor"
filename = 'models/' + model_name + '.sav'
grid_KN = pickle.load(open(filename, 'rb'))
predicted = grid_KN.predict(sm.X_test)
mae_KN = mean_absolute_error(sm.y_test, predicted)
forecast_KN = grid_KN.predict(sm.X_forecast)
res = ["KNeighbors Regressor", mae_KN, forecast_KN[-1], 'Time Series Cross Validation']
summary = add_result(summary, res)

model_name = "ExtraTrees Regressor"
filename = 'models/' + model_name + '.sav'
grid_ET = pickle.load(open(filename, 'rb'))
predicted = grid_ET.predict(sm.X_test)
mae_ET = mean_absolute_error(sm.y_test, predicted)
forecast_ET = grid_ET.predict(sm.X_forecast)
res = ["ExtraTrees Regressor", mae_ET, forecast_ET[-1], 'Time Series Cross Validation']
summary = add_result(summary, res)

models = [grid_RF, grid_DT, grid_GB, grid_AB, grid_ET, grid_KN]

modeldict = {
    "Random Forest Regressor": grid_RF,
    "Decision Tree Regressor": grid_DT,
    "Gradient Boosting Regressor": grid_GB,
    "AdaBoost Regressor": grid_AB,
    "ExtraTrees Regressor": grid_ET,
    "KNeighbors Regressor": grid_KN

}

# making options
years = [year for year in range(2010, 2021)]
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']
values['month_n'] = values['month'].apply(lambda x: months[x - 1])

# making alpha options for drop down menu
options_window = []
param_list = [a for a in np.arange(5, 35, 5)]
for param in param_list:
    option = dict(label=str(param), value=param)
    options_window.append(option)

# making alpha options for drop down menu
options_alpha = []
param_alpha = [a for a in np.arange(0.1, 1.1, 0.1)]
for alpha in param_alpha:
    option = dict(label='{0:.2f}'.format(alpha), value=alpha)
    options_alpha.append(option)

# making year options for drop down menu
options_year = []
option = dict(label='All', value=':')
options_year.append(option)
for year in years:
    option = dict(label=str(year), value=year)
    options_year.append(option)

# making month options for drop down menu
options_month = []
option = dict(label='All', value=':')
options_month.append(option)
for month in months:
    option = dict(label=month, value=month)
    options_month.append(option)

app.layout = html.Div([
    html.H1("Gold Price Prediction", style={"textAlign": "center"}),
    ##################
    html.Div([
        html.Div([
            html.Img(
                src="https://responsive.fxempire.com/cover/1230x820/webp-lossy-70.q50/_fxempire_/2020/06/Gold-Bars-2.jpg",
                className='two columns',
                style={
                    'height': '10%',
                    'width': '15%',
                    'float': 'right',
                    'position': 'relative',
                    'margin-top': 0,
                    'margin-right': 20,
                    'margin-bottom': 50
                },
            ),
            html.H3(children='Welcome to My  Gold Price Prediction Interactive Dashboard.',
                    className="nine columns"),

        ], className="row")
    ]),

    ##################
    html.Div([
        dcc.Markdown('''
 In this work, I have trained various time series and supervised learning models to
  predict the closing price of Gold. 
'''),
        html.P(children='',
               className="twelve columns"),

    ], className="row"),

    html.Div([
        dcc.Markdown('''

Disclaimer: Forecasting gold price is a very complex problem, since various non quantitative factors such as international political situation can affect it dramatically.
 Hence, the models used here are not capable of fully understand the market and should not be used for investment decisions.'''),
    ], className="row"),

    html.Br(),
    html.Spacer(),

    html.Div([

        html.Div(
            [
                html.P(children='', className='two columns'),
            ],
            className='two columns'
        ),
        html.Div(
            [
                html.Img(
                    src="",
                    className='four columns',
                    style={
                        'height': '10%',
                        'width': '10%',
                        'float': 'center',
                        'position': 'relative',
                        'margin-top': 0,
                        'margin-right': 0,
                        'margin-bottom': '2.5em'

                    }

                ),
            ],
            className='four columns'),

        html.Div(
            [
                html.P(children='', className='two columns'),
            ],
            className='two columns'

        ),

    ], className="row"),

    html.Div([
        dcc.Tabs(id="tabs", children=[

            # Tap 1
            dcc.Tab(label='Explanatory Data Analysis', children=[

                # Tap 1, Figure 1
                html.Div([
                    html.H1("How Gold Prices Are Fluctuated Over the Time",
                            style={'textAlign': 'center', 'padding-top': 5}),

                    dcc.Graph(id='priceHistory'),

                ], className="container"),

                # Tap 1, Figure 2
                html.Div([
                    html.H1("Visualising Trend",
                            style={'textAlign': 'center', 'padding-top': 5}),
                    html.H3("Select the years which you want to be included in the graph",
                            style={'textAlign': 'center', 'padding-top': 5}),
                    dcc.Dropdown(id='my-dropdown2', options=options_year,
                                 multi=True, value=[':'],
                                 style={"display": "block", "margin-left": "auto",
                                        "margin-right": "auto", "width": "80%"}),
                    dcc.Graph(id='year-wise_pattern')

                ], className="container"),

                # Tap 1, Figure 3
                html.Div([
                    html.H1("Visualising Trend Using Year-Wise Box Plot",
                            style={'textAlign': 'center', 'padding-top': 5}),
                    html.H3("Select the month which you want to be included in the graph",
                            style={'textAlign': 'center', 'padding-top': 5}),
                    dcc.Dropdown(id='my-dropdown3', options=options_month,
                                 multi=True, value=[':'],
                                 style={"display": "block", "margin-left": "auto",
                                        "margin-right": "auto", "width": "80%"}),
                    dcc.Graph(id='box_plot'),

                ], className="container"),

                # Tap 1, Figure 4
                html.Div([
                    html.H1("Visualising Trend Using Month-Wise Box Plot",
                            style={'textAlign': 'center', 'padding-top': 5}),
                    html.H3("Select the years which you want to be included in the graph",
                            style={'textAlign': 'center', 'padding-top': 5}),
                    dcc.Dropdown(id='my-dropdown4', options=options_year,
                                 multi=True, value=[':'],
                                 style={"display": "block", "margin-left": "auto",
                                        "margin-right": "auto", "width": "80%"}),
                    dcc.Graph(id='box_plot_month'),

                ], className="container")

            ]),  # end of Tab 1

            # Tap 2
            dcc.Tab(label='Time Series Prediction', children=[

                # Tap 2, Figure 1
                html.Div([
                    html.H1("Moving Average",
                            style={'textAlign': 'center', 'padding-top': 5}),
                    dcc.Dropdown(id='dropdown_T2_1', options=options_window,
                                 placeholder="Select window size",
                                 multi=True,
                                 style={"display": "block", "margin-left": "auto",
                                        "margin-right": "auto", "width": "80%"}),

                    dcc.RadioItems(id="radiob_T2_1", value="High",
                                   labelStyle={'display': 'inline-block', 'padding': 10},
                                   options=[{'label': "Show Higher and Lower Bounds", 'value': 'True'},
                                            {'label': "Don't Show Higher and Lower Bounds",
                                             'value': 'False'},
                                            ],
                                   style={'textAlign': "center", }),

                    dcc.Graph(id='Moving_Average'),

                ], className="container"),

                # Tap 2, Figure 2
                html.Div([
                    html.H1("Exponential Smoothing",
                            style={'textAlign': 'center', 'padding-top': 5}),
                    dcc.Dropdown(id='dropdown_T2_2', options=options_alpha,
                                 multi=True, placeholder="Select alpha",
                                 style={"display": "block", "margin-left": "auto",
                                        "margin-right": "auto", "width": "80%"}),

                    dcc.Graph(id='Exponential_Smoothing'),

                ], className="container"),

                # Tap 2, Figure 3
                html.Div([
                    html.H1("Double Exponential Smoothing",
                            style={'textAlign': 'center', 'padding-top': 5}),
                    dcc.Dropdown(id='dropdown_T2_3_1', options=options_alpha,
                                 placeholder="Select alpha",
                                 multi=False,
                                 style={"display": "block", "margin-left": "auto",
                                        "margin-right": "auto", "width": "80%"}),
                    dcc.Dropdown(id='dropdown_T2_3_2', options=options_alpha,
                                 placeholder="Select beta",
                                 multi=False,
                                 style={"display": "block", "margin-left": "auto",
                                        "margin-right": "auto", "width": "80%"}),

                    dcc.Graph(id='Double_Exponential_Smoothing'),

                ], className="container"),

            ]),  # end of Tab 2

            # Tap 3
            dcc.Tab(label='Machine Learning Prediction', children=[
                # Tap 3, Figure 1
                html.Div([
                    html.H1("Supervised Machine Learning Models",
                            style={'textAlign': 'center', 'padding-top': 5}),

                    html.P(
                        "These models predict the price of gold for the next four days. The models are trained and evaluated based on the same basis.",
                        style={'textAlign': 'center', 'padding-top': 5}),

                    html.H4("Comparing Supervised Machine Learning Models in Training and Testing Period",
                            style={'textAlign': 'center', 'padding-top': 5}),

                    dcc.Dropdown(id='dropdown_T3_1', options=[
                        # {'label': 'Linear Regression', 'value': 'Linear Regression'},
                        {'label': 'Random Forest Regressor', 'value': 'Random Forest Regressor'},
                        {'label': 'Decision Tree Regressor', 'value': 'Decision Tree Regressor'},
                        {'label': 'Gradient Boosting Regressor', 'value': 'Gradient Boosting Regressor'},
                        {'label': 'AdaBoost Regressor', 'value': 'AdaBoost Regressor'},
                        {'label': 'KNeighbors Regressor', 'value': 'KNeighbors Regressor'},
                        {'label': 'ExtraTrees Regressor', 'value': 'ExtraTrees Regressor'},
                    ],
                                 placeholder="Select Model",
                                 multi=False,
                                 style={"display": "block", "margin-left": "auto",
                                        "margin-right": "auto", "width": "80%"}),

                    dcc.Graph(id='models_prediction'),
                    # dcc.Graph(id='Linear_Regression_performance')

                ], className="container"),

                # tap 3, figure 2
                html.H1("Forecast of Supervised Machine Learning Models",
                        style={'textAlign': 'center', 'padding-top': 5}),

                dcc.Dropdown(id='dropdown_T3_2', options=[
                    {'label': 'All Models', 'value': ':'},
                    # {'label': 'Linear Regression', 'value': 'Linear Regression'},
                    {'label': 'Random Forest Regressor', 'value': 'Random Forest Regressor'},
                    {'label': 'Decision Tree Regressor', 'value': 'Decision Tree Regressor'},
                    {'label': 'Gradient Boosting Regressor', 'value': 'Gradient Boosting Regressor'},
                    {'label': 'AdaBoost Regressor', 'value': 'AdaBoost Regressor'},
                    {'label': 'KNeighbors Regressor', 'value': 'KNeighbors Regressor'},
                    {'label': 'ExtraTrees Regressor', 'value': 'ExtraTrees Regressor'},

                ],
                             placeholder="Select Model",
                             multi=True,
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "80%"}),
                dcc.Graph(id='Prediction_summary'),

                # tap 3, figure 3
                html.H1("Performance of Supervised Machine Learning Models",
                        style={'textAlign': 'center', 'padding-top': 5}),
                dcc.Graph(id='result_summary_graph'),

                html.H1(" ",
                        style={'textAlign': 'center', 'margin-bottom': '2.5em'}),

                # tap 3, figure 4
                html.H1("Summary of Supervised Machine Learning Models",
                        style={'textAlign': 'center', 'padding-top': 5}),

                dcc.Graph(id='result_summary'),

            ]),  # end of Tab 3

        ])
    ])
])


##** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** * TAB3 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *


# Figure T3_4
@app.callback(Output('result_summary', 'figure'),
              [Input('dropdown_T3_1', 'value')])
def update_graph(selected_dropdown):
    trace1 = []
    trace1.append(
        go.Table(
            header=dict(values=list(summary.columns),

                        align='center',
                        font=dict(color='black', size=16),
                        line_color='black',
                        fill_color='lightcyan',
                        height=50

                        ),
            cells=dict(values=[summary.Model, summary.MAE, summary.Prediction, summary.Validation],
                       line_color='gray',
                       fill_color='white',
                       font=dict(color='black', size=12),
                       height=50

                       )

        )
    )
    title = " "
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(
                  height=600,
                  title=title,
                  plot_bgcolor='rgba(0,0,0,0)')}

    return figure


# Figure T3_3
@app.callback(Output('result_summary_graph', 'figure'),
              [Input('dropdown_T3_1', 'value')])
def update_graph(selected_dropdown):
    trace1 = []
    trace1.append(
        go.Bar(
            x=summary.Model, y=summary.MAE,
            textposition='auto',
        ),
    )
    title = ''
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(
                  height=600,
                  title=title,
                  plot_bgcolor='rgba(0,0,0,0)',
                  yaxis={"title": "MAE"},
                  paper_bgcolor='rgba(0,0,0,0)',

              )}

    return figure


# Figure T3_2
@app.callback(Output('Prediction_summary', 'figure'),
              [Input('dropdown_T3_2', 'value')])
def update_graph(selected_dropdown):
    trace1 = []
    forecast = []

    selected_models = {}
    if (selected_dropdown != None):

        if ':' in selected_dropdown:
            selected_models = modeldict
        else:
            for selected in selected_dropdown:
                selected_models[selected] = modeldict[selected]

        for model_name in selected_models:
            model = modeldict[model_name]
            forecast = model.predict(sm.X_forecast)
            title = model_name + " Forecast"
            trace1.append(
                go.Scatter(x=sm.X_forecast.date, y=forecast, mode='lines+markers', opacity=0.9, name=title,
                           textposition='bottom center'))

    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=" ",
                                  xaxis={"title": "Date",
                                         'ticktext': sm.X_forecast.date
                                         },

                                  yaxis={"title": "Price (USD)"}, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)')}

    return figure


# Figure T3_1
@app.callback(Output('models_prediction', 'figure'),
              [Input('dropdown_T3_1', 'value')])
def update_graph(selected_dropdown):
    trace1 = []
    trace1.append(
        go.Scatter(x=sm.X_train.date, y=sm.y_train, mode='lines', opacity=0.6, name='Actual Train ',
                   textposition='bottom center'))

    trace1.append(
        go.Scatter(x=sm.X_test.date, y=sm.y_test, mode='lines', opacity=0.6, name='Actual Test',
                   textposition='bottom center'))

    prediction = []
    forecast = []
    title = ""
    mae = -1
    if (selected_dropdown != None):

        if selected_dropdown == 'Random Forest Regressor':
            prediction = grid_RF.predict(sm.X_test)
            forecast = grid_RF.predict(sm.X_forecast)
            mae = mae_RF
            title = " Training MAE: " + '{0:.2f}'.format(mae)

        elif selected_dropdown == 'Decision Tree Regressor':
            prediction = grid_DT.predict(sm.X_test)
            forecast = grid_DT.predict(sm.X_forecast)
            mae = mae_DT
            title = " Training MAE: " + '{0:.2f}'.format(mae)

        elif selected_dropdown == 'Gradient Boosting Regressor':
            prediction = grid_GB.predict(sm.X_test)
            forecast = grid_GB.predict(sm.X_forecast)
            mae = mae_GB
            title = " Training MAE: " + '{0:.2f}'.format(mae)

        elif selected_dropdown == 'AdaBoost Regressor':
            prediction = grid_AB.predict(sm.X_test)
            forecast = grid_AB.predict(sm.X_forecast)
            mae = mae_AB
            title = " Training MAE: " + '{0:.2f}'.format(mae)

        elif selected_dropdown == 'ExtraTrees Regressor':
            prediction = grid_ET.predict(sm.X_test)
            forecast = grid_ET.predict(sm.X_forecast)
            mae = mae_ET
            title = " Training MAE: " + '{0:.2f}'.format(mae)

        elif selected_dropdown == 'KNeighbors Regressor':
            prediction = grid_KN.predict(sm.X_test)
            forecast = grid_KN.predict(sm.X_forecast)
            mae = mae_KN
            title = " Training MAE: " + '{0:.2f}'.format(mae)

        name = str(selected_dropdown) + " Test"
        trace1.append(
            go.Scatter(x=sm.X_test.date, y=prediction, mode='lines', opacity=0.6, name=name,
                       textposition='bottom center'))

        name = str(selected_dropdown) + " Forecast"
        trace1.append(
            go.Scatter(x=sm.X_forecast.date, y=forecast, mode='lines', line=dict(color="#FF1493"), opacity=0.6,
                       name=name,
                       textposition='bottom center'))

    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=title,
                                  xaxis={"title": "Date",
                                         'rangeslider': {'visible': True},
                                         # 'ticktext': values.date
                                         },

                                  yaxis={"title": "Price (USD)"}, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)')}

    return figure


# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *TAB2 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
# Figure T2_3
@app.callback(Output('Double_Exponential_Smoothing', 'figure'),
              [Input('dropdown_T2_3_1', 'value'), Input('dropdown_T2_3_2', 'value')])
def update_graph(selected_dropdown1, selected_dropdown2):
    df = values["Gold"]

    trace1 = []
    trace1.append(
        go.Scatter(x=values.date, y=df, mode='lines', opacity=0.6, name='Actual values', textposition='bottom center'))

    if selected_dropdown1 != None and selected_dropdown2 != None:
        alpha = float(selected_dropdown1)
        beta = float(selected_dropdown2)

        model = Holt(np.asarray(np.asarray(df)))
        fit = model.fit(smoothing_level=alpha, smoothing_slope=beta)
        result = fit.fittedvalues

        name = 'Exponential Smoothing ' + '{0:.2f}'.format(alpha) + ',' + '{0:.2f}'.format(beta)
        trace1.append(
            go.Scatter(x=values.date, y=result, mode='lines', opacity=0.6, name=name, textposition='bottom center'))

    title = ""
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=title,
                                  xaxis={"title": "Date",
                                         'rangeslider': {'visible': True},
                                         },

                                  yaxis={"title": "Price (USD)"}, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)')}

    return figure


# Figure T2_2
@app.callback(Output('Exponential_Smoothing', 'figure'),
              [Input('dropdown_T2_2', 'value')])
def update_graph(selected_dropdown):
    df = values["Gold"]

    trace1 = []
    trace1.append(
        go.Scatter(x=values.date, y=df, mode='lines', opacity=0.6, name='Actual values', textposition='bottom center'))

    if selected_dropdown != None:
        for alpha in selected_dropdown:
            model = SimpleExpSmoothing(np.asarray(df))
            fit = model.fit(smoothing_level=alpha)
            result = fit.fittedvalues

            name = 'Exponential Smoothing ' + '{0:.2f}'.format(alpha)
            trace1.append(
                go.Scatter(x=values.date, y=result, mode='lines', opacity=0.6, name=name, textposition='bottom center'))

    title = ""
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=title,
                                  xaxis={"title": "Date",
                                         'rangeslider': {'visible': True},
                                         },

                                  yaxis={"title": "Price (USD)"}, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)')}

    return figure


# Figure T2_1
@app.callback(Output('Moving_Average', 'figure'),
              [Input('dropdown_T2_1', 'value'), Input("radiob_T2_1", "value")])
def update_graph(selected_dropdown, radioval):
    plot_intervals = False
    scale = 1.95
    df = values["Gold"]

    trace1 = []
    trace1.append(
        go.Scatter(x=values.date, y=df, mode='lines', opacity=0.6, name='Actual values', textposition='bottom center'))

    if selected_dropdown != None:
        for window in selected_dropdown:
            rolling_mean = df.rolling(window=window).mean()
            name = 'Moving Average ' + str(window)
            trace1.append(
                go.Scatter(x=values.date, y=rolling_mean, mode='lines', opacity=0.6, name=name,
                           textposition='bottom center'))

            if (radioval == 'True'):
                mae = mean_absolute_error(df[window:], rolling_mean[window:])
                deviation = np.std(df[window:] - rolling_mean[window:])

                name = 'Upper bound - Moving Average ' + str(window)
                lower_bound = rolling_mean - (mae + scale * deviation)
                trace1.append(
                    go.Scatter(x=values.date, y=lower_bound, mode='lines', opacity=0.6, name=name,
                               textposition='bottom center'))

                name = 'Lower bound - Moving Average ' + str(window)
                upper_bound = rolling_mean + (mae + scale * deviation)
                trace1.append(
                    go.Scatter(x=values.date, y=upper_bound, mode='lines', opacity=0.6, name=name,
                               textposition='bottom center'))
    title = ""
    # title = "MAE for window " + str(window) + " is " + str(mae)
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=title,
                                  xaxis={"title": "Date",
                                         'rangeslider': {'visible': True},
                                         },

                                  yaxis={"title": "Price (USD)"}, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)')}

    return figure


# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *TAB1 ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
# Figure T1_4
@app.callback(Output('box_plot_month', 'figure'),
              [Input('my-dropdown4', 'value')])
def update_graph(selected_dropdown):
    N = 12
    c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, N)]

    if ':' in selected_dropdown:
        selected_dropdown = years

    trace1 = []
    df = values.loc[values.year.isin(selected_dropdown), :]
    y = df['Gold']
    x = df['month_n']
    trace1.append(
        go.Box(x=x, y=y, boxpoints='all', jitter=0.5, whiskerwidth=0.2, marker_size=2, line_width=1)
    )

    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title="",
                                  xaxis={"title": "Month",
                                         'rangeslider': {'visible': True},
                                         'ticktext': months
                                         },
                                  yaxis={"title": "Price (USD)"}, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)')}

    return figure


# Figure T1_3
@app.callback(Output('box_plot', 'figure'),
              [Input('my-dropdown3', 'value')])
def update_graph(selected_dropdown):
    N = 12
    c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, N)]
    if ':' in selected_dropdown:
        selected_dropdown = months

    trace1 = []

    df = values.loc[values.month_n.isin(selected_dropdown), :]
    y = df['Gold']
    x = df['year']
    trace1.append(
        go.Box(x=x, y=y, boxpoints='all', jitter=0.5, whiskerwidth=0.2, marker_size=2, line_width=1)
    )

    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title="",
                                  xaxis={"title": "Year",
                                         'rangeslider': {'visible': True},
                                         'ticktext': months
                                         },
                                  yaxis={"title": "Price (USD)"}, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)')}

    return figure


# Figure T1_2
@app.callback(Output('year-wise_pattern', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown):
    df_pivot = pd.pivot_table(df, values='Gold', index='month', columns='year', aggfunc='mean').sort_values(
        by="month")

    trace1 = []

    df_pivot['m'] = df_pivot.index.map(lambda x: months[x - 1])
    if ':' in selected_dropdown:
        selected_dropdown = years

    for year in selected_dropdown:
        trace1.append(
            go.Scatter(x=df_pivot['m'], y=df_pivot[year], mode='lines', opacity=0.6, name=year,
                       textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title="",
                                  xaxis={"title": "Month",
                                         'rangeslider': {'visible': True},
                                         'ticktext': months
                                         },

                                  yaxis={"title": "Price (USD)"}, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)')}

    return figure


# Figure T1_1
@app.callback(Output('priceHistory', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown):
    trace1 = []
    trace1.append(
        go.Scatter(x=df.index, y=df["Gold"], mode='lines', opacity=0.6, name='Gold', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title="",
                                  xaxis={"title": "Date",
                                         'rangeselector': {
                                             'buttons': list([
                                                 dict(count=1, label="Last Month", step="month", stepmode="backward"),
                                                 dict(count=6, label="Last 6 Months", step="month",
                                                      stepmode="backward"),
                                                 dict(count=1, label="Since Jan 2020", step="year", stepmode="todate"),
                                                 dict(count=1, label="Last Year", step="year", stepmode="backward"),
                                                 dict(count=5, label="Last 5 Years", step="year", stepmode="backward"),
                                                 dict(step="all", label="Since Jan 2010")
                                             ])
                                         },
                                         'rangeslider': {'visible': True}, 'type': 'date'},
                                  yaxis={"title": "Price (USD)"}, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)')}
    return figure


if __name__ == '__main__':
    application.run(debug=False, port=8080)
