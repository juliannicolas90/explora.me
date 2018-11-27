import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt

import dash_table

import plotly.plotly as py
from plotly import tools
import plotly.graph_objs as go

import pandas as pd

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions']=True
app.scripts.config.serve_locally = True

app.layout = html.Div([
    html.H1("explora.me", style={'textAlign': 'center', 'font-family': 'Courier New'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])


    dropdown1 = dcc.Dropdown(
    options=[{'label': col, 'value': col} for col in df.columns if check_categorical(df, col)],
    id='dropdown1',
    placeholder='Select a variable defining the groups'
    )    

    dropdown2 = dcc.Dropdown(
    options=[{'label': col, 'value': col} for col in df.columns],
    id='dropdown2',
    placeholder='Select a variable to evaluate'
    )   

    df_table = dash_table.DataTable(
        id='dataframe',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict("rows"),
        style_cell={
        # all three widths are needed
        'whiteSpace': 'no-wrap',
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
        },
        style_table={
        'maxHeight': '300',
        'overflowY': 'scroll'
        },
        editable=True,
        css=[{
        'selector': '.dash-cell div.dash-cell-value',
        'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
        }],
        )

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # Use the DataTable prototype component:
        # github.com/plotly/datatable-experiments
        df_table,

        html.Hr(),  # horizontal line

        dropdown1,
        dropdown2,

        html.Div(id='responder'),

        # For debugging, display the raw contents provided by the web browser
 #       html.Div('Raw Content'),
 #       html.Pre(contents[0:200] + '...', style={
 #           'whiteSpace': 'pre-wrap',
 #           'wordBreak': 'break-all'
 #       })
    ])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(
    Output(component_id='responder', component_property='children'),
    [Input(component_id='dropdown1', component_property='value'), Input(component_id='dropdown2', component_property='value'), Input('dataframe', 'data'), Input('dataframe', 'columns')]
)
def update_output_div(var1, var2, rows, columns):
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])

    if check_categorical(df, var2):
        ##Check for expected frequency to use Fisher!!
        return [print_chi2(df, var1, var2), draw_categorical(df, var1, var2)]

    elif is_numeric_dtype(df[var2]):
        return [print_numeric(df, var1, var2), draw_numeric(df, var1, var2)]

from scipy.stats import f_oneway, ttest_ind, mannwhitneyu, kruskal, chisquare, fisher_exact, chi2_contingency
from statsmodels.stats.diagnostic import kstest_normal


def print_chi2(df, group_variable, variable):
    ct = pd.crosstab(df[variable], df[group_variable])
    chi2, p, dof, expected_freq = chi2_contingency(ct)
    ct.insert(0, 'Level/Group', pd.Series(df[variable].dropna().unique(), index=ct.index))
    ct_table = dash_table.DataTable(
        id='ct_table',
        columns=[{"name": i, "id": i} for i in ct.columns],
        data=ct.to_dict("rows"))
    return html.Div([ct_table, html.Div("The chi2 is {}, with a p of {}.".format(chi2, p))])

def print_numeric(df, group_variable, variable):
    if len(df[group_variable].unique()) > 2:
        if check_normality(df, variable):
            return print_anova(df, group_variable, variable)
        else:
            return print_kruskal(df, group_variable, variable)
    else:
        if check_normality(df, variable):
            return print_ttest(df, group_variable, variable)
        else:
            return print_mannwhitney(df, group_variable, variable)

def check_normality(df, variable, alpha=0.05):
    ##use only KS or Anderson Starling also?
    ks, p = kstest_normal(df[variable])
    if p>alpha:
        return True
    return False

def print_ttest(df, group_variable, variable):
    groups = [df[df[group_variable]==group][variable] for group in df[group_variable].unique()]
    t, p = ttest_ind(*groups, nan_policy="omit")
    p = "<0.0001" if p<0.0001 else round(p, 4)
    return html.Div("The t is {}, with a p of {}.".format(t, p))

def print_anova(df, group_variable, variable):
    groups = [df[df[group_variable]==group][variable].dropna() for group in df[group_variable].unique()]
    F, p = f_oneway(*groups)
    p = "<0.0001" if p<0.0001 else round(p, 4)
    return html.Div("The F is {}, with a p of {}.".format(F, p))

def print_mannwhitney(df, group_variable, variable):
    groups = [df[df[group_variable]==group][variable].dropna() for group in df[group_variable].unique()]
    u, p = mannwhitneyu(*groups)
    p = "<0.0001" if p<0.0001 else round(p, 4)
    return html.Div("The u is {}, with a p of {}.".format(u, p))    

def print_kruskal(df, group_variable, variable):
    groups = [df[df[group_variable]==group][variable].dropna() for group in df[group_variable].unique()]
    H, p = kruskal(*groups)
    p = "<0.0001" if p<0.0001 else round(p, 4)
    return html.Div("The H is {}, with a p of {}.".format(H, p))    

def draw_categorical(df, group_variable, variable):
    data = [go.Bar(
        x=df[df[variable]==i][group_variable].value_counts().keys(),
        y=df[df[variable]==i][group_variable].value_counts().values,
        name=str(i)) for i in df[variable].unique()]
    graph = dcc.Graph(
        id="plot",
        figure={
        'data': data,
        'layout': go.Layout(    
                xaxis={'title': group_variable},
                yaxis={'title': "Count"},
                margin={'l': 100, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 1, 'y': 1},
                hovermode=  'closest'
                )}
        )
    return graph


def draw_numeric(df, group_variable, variable):
    traces = [go.Histogram(x=df[df[group_variable]==i][variable], name=str(i)) for i in df[group_variable].unique()]
    fig = tools.make_subplots(cols=len(traces))
    for i in range(len(traces)):
        fig.append_trace(traces[i], 1, i+1)

    graph = dcc.Graph(
        id="plot",
        figure={
        'data': fig,
        'layout': go.Layout(    
                xaxis={'title': variable},
                yaxis={'title': "Count"},
                margin={'l': 100, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 1, 'y': 1},
                hovermode=  'closest'
                )            
        }
        )
    return graph

def check_categorical(dataset, col):
    if ((len(dataset[col].value_counts())<=2 or is_string_dtype(dataset[col])) and dataset[col].value_counts().iloc[0]!=len(dataset)):
        return True
    return False

if __name__ == '__main__':
    app.run_server(debug=True)