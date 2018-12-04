import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import dash_table

import plotly.plotly as py
from plotly import tools
import plotly.graph_objs as go

import numpy as np
import pandas as pd

from flask import send_file, session, request, abort
import io

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import flask

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'style/styles.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

year = datetime.datetime.now().year


from flask_sslify import SSLify
sslify = SSLify(server)

app.config['suppress_callback_exceptions']=True
app.title = "explora.me"
app.server.secret_key = "asndiopasmdasdnasdosanjdp"
#app.scripts.config.serve_locally = True



app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
        </footer>
    </body>
</html>
'''

upload_component = dcc.Upload(
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
        }, multiple=True)

##Index page with upload component
app.layout = html.Div([
    html.H1("explora.me", style={'textAlign': 'center'}, className="logo"),
    upload_component,
    html.Div(id='output-data-upload'),
    html.Div([
        html.Div(id='analyze_table', className="six columns", style={'padding': '5px'}),
        html.Div(id='analyze_table_multiple', className="six columns", style={'padding': '5px', 'paddingBottom': "20px"}),
    ]),
    html.Div("No data is stored by explora.me. The statistics displayed here are for exploratory use only. Statistical analysis should be performed by a statistician. Copyright © {} by Julián Nicolás Acosta.".format(year), className="footer")
])

##Function to read the uploaded files (excel or csv)

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
    df_table = dash_table.DataTable(
        id='dataframe',
        columns=[{"name": i, "id": i,'deletable': True,'editable_name': True} for i in df.columns],
        data=df.to_dict("rows"),
        row_deletable=True,
        style_cell={
        'color': 'black',
        'minWidth': '200px',
        },
        style_table={
        'maxHeight': '300',
        'overflowY': 'scroll'
        },
        editable=True,
        n_fixed_rows=1 
        )

    return html.Div([
        html.H5(filename[:filename.find(".")], style={'textAlign': 'center'}),
        df_table,
    ])

##Callback to show the data 
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

##Callbacks to run when the data is uploaded; single or multiple variable analysis

@app.callback(Output(component_id='analyze_table', component_property='children'),
            [Input('dataframe', 'data'), Input('dataframe', 'columns')])
def analyze_table(rows, columns):
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    dropdown1 = dcc.Dropdown(
    options=[{'label': col, 'value': col} for col in df.columns if check_categorical(df, col)],
    id='dropdown1',
    placeholder='Select a variable defining the groups',
    className="dropdown"
    )    
    dropdown2 = dcc.Dropdown(
    options=[{'label': col, 'value': col} for col in df.columns],
    id='dropdown2',
    placeholder='Select a variable to evaluate',
    className="dropdown"
    )   
    return [
        html.Hr(),  # horizontal line

        html.H4("Single Variable Analysis", style=dict(textAlign="center")),
        html.H6("Choose a Group Variable:", style=dict(textAlign="center")),
        dropdown1,
        html.H6("Choose a Analysis Variable:", style=dict(textAlign="center")),
        dropdown2,

        html.Hr(),

        html.Div(id='responder'),
        ]

@app.callback(Output(component_id='analyze_table_multiple', component_property='children'),
            [Input('dataframe', 'data'), Input('dataframe', 'columns')])
def analyze_table_multiple(rows, columns):
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    dropdown1 = dcc.Dropdown(
    options=[{'label': col, 'value': col} for col in df.columns if check_categorical(df, col)],
    id='dropdown1_multiple',
    placeholder='Select a variable defining the groups',
    className="dropdown"
    )    
    dropdown2 = dcc.Dropdown(
    options=[{'label': col, 'value': col} for col in df.columns if columns],
    multi=True,
    id='dropdown2_multiple',
    placeholder='Select a variable to evaluate',
    className="dropdown"
    )   
    return [
        html.Hr(),  # horizontal line
        html.H4("Multiple Variable Analysis", style=dict(textAlign="center")),
        html.H6("Choose a Group Variable:", style=dict(textAlign="center")),
        dropdown1,
        html.H6("Choose a Analysis Variable:", style=dict(textAlign="center")),
        dropdown2,

        html.Hr(),

        html.Div(id='responder_multiple'),
        ]

##Callback for when the user analyses data for single or multiple variables

@app.callback(
    Output(component_id='responder', component_property='children'),
    [Input(component_id='dropdown1', component_property='value'), Input(component_id='dropdown2', component_property='value'), Input('dataframe', 'data'), Input('dataframe', 'columns')]
)
def compare_single_variable(var1, var2, rows, columns):
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    if not var1 or not var2:
        return
    if check_categorical(df, var2):
        if check_for_fisher(df, var1, var2):
            return[print_fisher(df, var1, var2), draw_categorical(df, var1, var2)]
        else:
            return [print_chi2(df, var1, var2), draw_categorical(df, var1, var2)]

    elif is_numeric_dtype(df[var2]):
        return [print_summary(df, var1, var2), print_numeric(df, var1, var2), draw_numeric(df, var1, var2)]


@app.callback(
    Output(component_id='responder_multiple', component_property='children'),
    [Input(component_id='dropdown1_multiple', component_property='value'), Input(component_id='dropdown2_multiple', component_property='value'), Input('dataframe', 'data'), Input('dataframe', 'columns')]
)
def compare_multiple_variables(var1, vars, rows, columns):
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    if not var1 or not vars:
        return
    group_names = list(map(str,df[var1].unique()))
    columns = ['Variable'] + list(group_names) + ['p value']
    rows = []
    for var in vars:
        if check_categorical(df, var):
            ##Calculate percent of each category
            ##Calculate Chi2
            var_rows = []
            row_1 = {group_name: "" for group_name in group_names}
            row_1['Variable'] = var
            p = calculate_fisher(df, var1, var) if check_for_fisher(df, var1, var) else calculate_chi2(df, var1, var)
            row_1['p value'] = str(round(p,4)) if p>= 0.0001 else "<0.0001"
            var_rows.append(row_1)
            categories = df[var].unique()
            groups = get_groups(df, var1, var, dropna=False)
            for cat in categories:
                row = {'Variable': " "+str(cat)}
                for i in range(len(groups)):
                    cnt = groups[i].value_counts()[cat] if cat in groups[i].unique() else 0
                    percent = round(cnt/len(groups[i])*100,2)
                    row[group_names[i]] = "{} ({}%)".format(cnt, percent)
                    row['p value'] = ""
                var_rows.append(row)
            rows += var_rows
        elif is_numeric_dtype(df[var]):
            groups = get_groups(df, var1, var, dropna=True)
            ##Calculate means + std and median + IQR
            ##Calculate t-test, MW, ANOVA o Kruskel
            var_rows = []
            row_1 = {group_name: "" for group_name in group_names}
            row_1['Variable'] = var
            p = get_p_numerical(df, var1, var)
            row_1['p value'] = str(round(p,4)) if p>= 0.0001 else "<0.0001"
            row_mean = {group_names[i]: "{} +- {}".format(round(groups[i].mean(),2), round(groups[i].std(),2)) for i in range(len(groups))}
            row_mean['Variable'] = "Mean +- Std"
            row_mean['p value'] = ""
            row_median = {group_names[i]: "{} ({}-{})".format(round(groups[i].median(),2), groups[i].quantile(0.25), groups[i].quantile(0.75)) for i in range(len(groups))}
            row_median['Variable'] = "Median (IQR)"
            row_median['p value'] = ""
            var_rows.append(row_1)
            var_rows.append(row_mean)
            var_rows.append(row_median)
            rows += var_rows
    df_table = pd.DataFrame(rows, columns=[c for c in columns])
    ##Create DataTable
    multiple_table = generate_table(df_table)
    session['multiple_table'] = df_table.to_json(orient="split")
    return [multiple_table, html.A("Save", className="button", href="/download_multiple/")]

TABLE_STYLE = {
    "backgroundColor": "White",
    "borderRadius": "5px",
    "color": "black",
}

def generate_table(dataframe):
    return html.Div([html.Table(
        # Header
        [html.Tr([html.Th(col, style={"padding": "5px"}) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col], style={"padding": "5px"}) for col in dataframe.columns
        ]) for i in range(len(dataframe))],
        className="twelve columns",
        style=TABLE_STYLE)], style={'overflow-x': "auto"}
    )

@app.server.route("/download_multiple/")
def download_multiple():
    if not request.referrer:
        abort(404)
    data = session.get('multiple_table', None)
    df = pd.read_json(data, orient="split")
    strIO = io.BytesIO()
    excel_writer = pd.ExcelWriter(strIO, engine="xlsxwriter")
    df.to_excel(excel_writer, index=False, sheet_name="Sheet 1")
    excel_writer.save()
    excel_data = strIO.getvalue()
    strIO.seek(0)
    return send_file(strIO,
        attachment_filename='multiple_table.xlsx',
        as_attachment=True)

###Statistical tests and graphics for single variable

from scipy.stats import f_oneway, ttest_ind, mannwhitneyu, kruskal, chisquare, fisher_exact, chi2_contingency
from scipy.stats.contingency import expected_freq
from statsmodels.stats.diagnostic import kstest_normal


GLOBAL_N_FOR_FISHER = 30
GLOBAL_EXPECTED_FREQ = 5


def get_p_numerical(df, group_variable, var):
    """Get dataframe, group variable and a numerical variable and return p using
    the corresponding statistical test according to assumptions."""
    groups = get_groups(df,group_variable, var)
    if len(df[group_variable].unique()) > 2:
        if check_normality(df, var):
            _, p = f_oneway(*groups)
        else:
            _, p = kruskal(*groups)
    else:
        if check_normality(df, var):
            _, p = ttest_ind(*groups)
        else:
            _, p =  mannwhitneyu(*groups)
    return p 

def check_for_fisher(df, var1, var2):
    exp_freq = expected_freq(pd.crosstab(df[var1], df[var2]))
    if exp_freq.shape != (2,2):
        return False
    if (exp_freq<=GLOBAL_EXPECTED_FREQ).any(axis=None) or len(df)<=GLOBAL_N_FOR_FISHER: 
        return True
    return False

def calculate_chi2(df, group_variable, variable):
    if len(df[variable].dropna().unique())==1:
        df[variable] = df[variable].fillna(0)
    ct = pd.crosstab(df[variable], df[group_variable])
    _,p,_,_ = chi2_contingency(ct)
    return p

def calculate_fisher(df, group_variable, variable):
    if len(df[variable].dropna().unique())==1:
        df[variable] = df[variable].fillna(0)
    ct = pd.crosstab(df[variable], df[group_variable])
    oddsr, p = fisher_exact(ct)
    return p

def print_chi2(df, group_variable, variable):
    if len(df[variable].dropna().unique())==1:
        df[variable] = df[variable].fillna(0)
    ct = pd.crosstab(df[variable], df[group_variable])
    chi2, p, dof, expected_freq = chi2_contingency(ct)
    ct.insert(0, '{}/{}'.format(variable, group_variable), pd.Series(df[variable].dropna().unique(), index=ct.index))
    ct_table = generate_table(ct)
    return html.Div([ct_table, html.Div("The chi2 is {}, with a p of {}.".format(chi2, p))])

def print_fisher(df, group_variable, variable):
    if len(df[variable].dropna().unique())==1:
        df[variable] = df[variable].fillna(0)
    ct = pd.crosstab(df[variable], df[group_variable])
    oddsr, p = fisher_exact(ct)
    ct.insert(0, '{}/{}'.format(variable, group_variable), pd.Series(df[variable].dropna().unique(), index=ct.index))
    ct_table = generate_table(ct)
    return html.Div([ct_table, html.Div("The Fisher Exact test for differences in proportions gives a p of {}.".format(p))])

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

def print_summary(df, group_variable, variable):
    group_names = df[group_variable].unique()
    summ_tb = pd.concat([df[df[group_variable]==group][variable].describe() for group in group_names], axis=1, keys=group_names)
    summ_tb.insert(0, 'Statistics for {}'.format(variable), summ_tb.index)
    return generate_table(summ_tb)

def check_normality(df, variable, alpha=0.05):
    ##use only KS or Anderson Starling also?
    ks, p = kstest_normal(df[variable])
    if p>alpha:
        return True
    return False

def get_groups(df, group_variable, variable, dropna=True):
    if dropna:
        return [df[df[group_variable]==group][variable].dropna() for group in df[group_variable].unique()]
    else:
        return [df[df[group_variable]==group][variable] for group in df[group_variable].unique()]

def print_ttest(df, group_variable, variable):
    groups = get_groups(df, group_variable, variable)
    t, p = ttest_ind(*groups, nan_policy="omit")
    p = "<0.0001" if p<0.0001 else round(p, 4)
    return html.Div("{} is normally distributed according to Kolmogorov-Smirnov test. The t statistic for comparing means is {}, with a p of {}.".format(variable, t, p))

def print_anova(df, group_variable, variable):
    groups = get_groups(df, group_variable, variable)
    F, p = f_oneway(*groups)
    p = "<0.0001" if p<0.0001 else round(p, 4)
    return html.Div("{} is normally distributed according to Kolmogorov-Smirnov test. The ANOVA (F statistic) for comparing means is {}, with a p of {}.".format(variable, F, p))

def print_mannwhitney(df, group_variable, variable):
    groups = get_groups(df, group_variable, variable)
    u, p = mannwhitneyu(*groups)
    p = "<0.0001" if p<0.0001 else round(p, 4)
    return html.Div("{} is not normally distributed according to Kolmogorov-Smirnov test. The u (Mann-Whitney) statistic for comparing distributions is {}, with a p of {}.".format(variable, u, p))    

def print_kruskal(df, group_variable, variable):
    groups = get_groups(df, group_variable, variable)
    H, p = kruskal(*groups)
    p = "<0.0001" if p<0.0001 else round(p, 4)
    return html.Div("{} is not normally distributed according to Kolmogorov-Smirnov test. The H (Kruskal-Wallis) statistic for comparing distributions is {}, with a p of {}.".format(variable, H, p))    

def draw_categorical(df, group_variable, variable, percent=True):
    if percent:
        categories = df[group_variable].unique()
        data = [go.Bar(
            x = categories,
            y=df[df[variable]==i][group_variable].value_counts()[categories]/df[group_variable].value_counts()[categories]*100,
            name=str(i)) for i in df[variable].unique()]
    else:
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
                yaxis={'title': "Percent" if percent else "Count"},
                margin={'l': 100, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 1, 'y': 1},
                hovermode=  'closest'
                )}
        )
    return graph

def create_graph(traces, id, xtitle="", ytitle=""):
    fig = tools.make_subplots()
    for i in range(len(traces)):
        fig.append_trace(traces[i], 1, 1)

    graph = dcc.Graph(
        id=id,
        figure={
        'data': fig,
        'layout': go.Layout(    
                xaxis={'title': xtitle},
                yaxis={'title': "Count"},
                margin={'l': 100, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 1, 'y': 1},
                hovermode=  'closest'
                )            
        },
        )    
    return graph

def draw_numeric(df, group_variable, variable):
    return html.Div([draw_histograms(df, group_variable, variable), draw_boxplots(df, group_variable, variable)],className='graph_container')

def draw_histograms(df, group_variable, variable):
    groups = df[group_variable].unique()
    traces = [go.Histogram(x=df[df[group_variable]==i][variable], name=str(i), legendgroup=str(i)) for i in groups]
    return html.Div(create_graph(traces, "histogram", xtitle=variable))#, className="six columns")

def draw_boxplots(df, group_variable, variable):
    groups = df[group_variable].unique()
    traces = [go.Box(y=df[df[group_variable]==i][variable], name=str(i), legendgroup=str(i)) for i in groups]
    return html.Div(create_graph(traces, "boxplot", xtitle=variable))#, className="six columns")

def check_categorical(dataset, col):
    if ((len(dataset[col].value_counts())<=2 or is_string_dtype(dataset[col])) and dataset[col].value_counts().iloc[0]!=len(dataset)):
        return True
    return False

if __name__ == '__main__':
    app.run_server(debug=False)