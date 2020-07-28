import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
import utils
import plotly.express as px
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H3(children='FCR_Model'),
    html.H6('lambda_i'),
    dcc.Slider(id='lambda_i',
               min=0.001,
               max=5,
               step=0.01,
               value=0.1),
    html.H6('lambda_f'),
    dcc.Slider(id='lambda_f',
               min=0.001,
               max=5,
               step=0.01,
               value=0.1),
    dcc.Graph(id='shows'),
    dcc.Graph(id='test',
              figure=go.Figure(
                  data=[go.Bar(y=[2, 1, 3])],
                  layout_title_text="A Figure Displayed with fig.show()"
              )
              )
])


@app.callback(
    Output('shows', 'figure'),
    [Input('lambda_i', 'value'),
     Input('lambda_f', 'value')])
def figure_update(lambda_i, lambda_f):
    lambda_list = [lambda_i, lambda_f]
    model1 = utils.FCR_bac(lambda_list)
    model1.integ_sigma()
    fig = px.line(x=model1.t, y=model1.j_r, labels={'J_Rb'})
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)