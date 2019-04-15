import base64
import datetime
import io
import os 
import numpy as np
import time
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

import pandas as pd

import process_image


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,hot_reload=True)
app.loaded_model = None
server = app.server
# app.css.config.serve_locally = True
# app.scripts.config.serve_locally = True
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1">

        <title>Plant Classifier</title>

        {%favicon%}
        <link href="https://codepen.io/chriddyp/pen/bWLwgP.css" rel="stylesheet">
        <link href="https://codepen.io/chriddyp/pen/brPBPO.css" rel="stylesheet">
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">
       
        {%css%}
    </head>
    <body>
        
        {%app_entry%}
        <footer id="footer">
            {%config%}
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js" integrity="sha384-Tc5IQib027qvyjSMfHjOMaLkfuWVxZxUPnCJA7l2mCWNIpG9mGCD8wGNIcPD7Txa" crossorigin="anonymous"></script>
            {%scripts%}
     
        </footer>
    </body>
</html>
'''


app.layout = html.Div([
    html.H2("Plant Leaf Classifier using Deep Learning", className="text-center"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        accept="image/*",
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
], className="container-fluid")

def parse_contents(contents, filename, date):

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    
    timestamp = int(datetime.datetime.now().timestamp())
    # filename = "{}_{}".format(str(timestamp), filename) 
    ext = filename.split(".")[-1]
    newfilename = "image."+ext #"{}_{}".format(str(timestamp), filename) 
    filepath = os.path.join("assets","images", newfilename )
    print(filepath)
    with open(filepath, 'wb') as f:
        f.write(decoded)

    if not app.loaded_model:
        # Get index to class mapping
        print("Loading model....")
        loaded_model, class_to_idx = process_image.load_checkpoint('assets/plants9615_checkpoint.pth')
        idx_to_class = { v : k for k,v in class_to_idx.items()}

    try:
        start = time.time()
        p, c = process_image.predict(filepath, loaded_model, idx_to_class)
        end = time.time()

        running_time = end - start
        print(p,c)
        y_pos = np.arange(len(p))
        print(y_pos)
        
        return html.Div([
           
            # HTML images accept base64 encoded strings in the same format
            # that is supplied by the upload
                
            html.Div(
                [
                    
                    html.Div([
                        html.H5("Prediction time: %s secs"%(round(running_time, 2))),
                        html.Img(src=contents),
                        html.Label("Filename: "+filename),
                        html.Label("Date modified:  " + str(datetime.datetime.fromtimestamp(date))),
                    
                    ]),
                ],
                className="text-center col-md-4", style={'top':'20px'}
            ),
            
            # html.Hr(),
            
            html.Div([
                
                dcc.Graph(
                    figure={
                    'data': [go.Bar(x=p, 
                            y=y_pos,
                            text=p,
                            textposition = 'inside',
                            orientation='h')],

                    'layout': go.Layout(
                        margin = go.layout.Margin(
            l = 350,
        ),
                        yaxis = dict(autorange= 'reversed', tickvals = y_pos, ticktext = c),
                        title = "Five Top Predictions and Probabilities"
                    )
                            }
                )
            ], className="text-center col-md-8")
        ], className="row", style={'padding': '10px'})
    except Exception as ex:
        print(ex)
        return html.Div('An error occurred while processing this image')



@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    start = time.time()
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        end = time.time()
        children.insert(0, 
            html.H2("Total running time: %s secs"%(round(end - start, 2))),
        )
        return children



if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=80)