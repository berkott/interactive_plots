import os
import json
import itertools
import numpy as np

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output,callback
import dash_bootstrap_components as dbc

MODEL_DIR = "models/good_sweep_1"

# === Plotting ===
def plot_differences_to_models_mu(model_save_path):
    # === Load Data ===
    plot_x_data = []
    plot_y_data = {
        "d_m_bd": {"name": "Disc Prior: Model-Bayes Disc", "data": []}, 
        "d_m_bc": {"name": "Disc Prior: Model-Bayes Cont", "data": []},
        "d_m_js": {"name": "Disc Prior: Model-James Stein", "data": []},
        "d_m_t": {"name": "Disc Prior: Model-True", "data": []},
        "d_bd_t": {"name": "Disc Prior: Bayes Disc-True", "data": []},
        "d_bc_t": {"name": "Disc Prior: Bayes Cont-True", "data": []},
        "d_js_t": {"name": "Disc Prior: James Stein-True", "data": []},
        "c_m_bd": {"name": "Cont Prior: Model-Bayes Disc", "data": []},
        "c_m_bc": {"name": "Cont Prior: Model-Bayes Cont", "data": []},
        "c_m_js": {"name": "Cont Prior: Model-James Stein", "data": []},
        "c_m_t": {"name": "Cont Prior: Model-True", "data": []},
        "c_bd_t": {"name": "Cont Prior: Bayes Disc-True", "data": []},
        "c_bc_t": {"name": "Cont Prior: Bayes Cont-True", "data": []},
        "c_js_t": {"name": "Cont Prior: James Stein-True", "data": []}
    }

    try:
        with open(os.path.join(model_save_path, "plot_data.json"), "r") as f:
                plot_data_json = json.load(f)
    except:
        print(f"Could not find plot_data.json in {model_save_path}")
        return None

    for epoch in plot_data_json.keys():
        if int(epoch) % 250 != 0:
            continue

        plot_x_data.append(int(epoch))

        for plot_y_data_key in plot_y_data.keys():
            if plot_y_data_key in plot_data_json[epoch].keys():
                plot_y_data[plot_y_data_key]["data"].append(plot_data_json[epoch][plot_y_data_key])

    plot_x_data = np.array(plot_x_data)
    sort_order = np.argsort(plot_x_data)

    for plot_y_data_key in plot_y_data.keys():
        plot_y_data[plot_y_data_key]["data"] = np.array(plot_y_data[plot_y_data_key]["data"])[sort_order]

    # === Plotting ===
    fig_dict = {
        "data": [],
        "layout": {}
    }

    model_signature = model_save_path.split('/')[-1]
    model_name = model_signature.split('-')[0]
    model_params = {model_arg.split('=')[0]: model_arg.split('=')[1] for model_arg in model_signature.split('-')[1:]}
    keys_to_show = ("pretraining_mu_tasks", "hid_dim", "layer", "head", "default_sigma")
    model_params = {key: model_params[key] for key in model_params if key in keys_to_show}

    fig_dict["layout"]["title"] = {"text": f"Differences mu {model_name}<br><sup>{str(model_params)}</sup>"}
    fig_dict["layout"]["xaxis"] = {"title": "Epochs"}
    fig_dict["layout"]["yaxis"] = {"range": [0, 2], "title": "Normalized MSE"}
    fig_dict["layout"]["height"] = 1000
    fig_dict["layout"]["legend"] = {"x": 0.8, "y": 0.95}
    fig_dict["layout"]["font"] = {"size": 18}
    fig_dict["data"] = [
        {
            "x": plot_x_data,
            "y": plot_y_data[plot_y_data_key]["data"],
            "name": plot_y_data[plot_y_data_key]["name"],
            "mode": "lines"
            # "mode": "lines+markers"
        }
        for plot_y_data_key in plot_y_data.keys()
    ]
    fig = go.Figure(fig_dict)

    return fig

# === Dash Setup ===
def get_settings_table():
    table_header = [
        html.Thead(html.Tr([html.Th("Parameter"), html.Th("Value")]))
    ]
    row1 = html.Tr([html.Td("Mus"), html.Td("See meeting notes 10/23")])
    table_body = [html.Tbody([row1])]
    return table_header + table_body

@callback(
    Output('mu-plot-container', 'children'),
    Input('select-mu-dropdown', 'value'),
    Input('select-hidden-dim-dropdown', 'value')
)
def update_dropdowns(mu_dropdown_value, hidden_dim_dropdown_value):
    if mu_dropdown_value == "continuous":
        return []
    
    mu_dirs = [dir.path for dir in os.scandir(MODEL_DIR) 
               if dir.is_dir() and 
               dir.path.split('/')[-1].split('-')[1][14:] == "mu" and 
               dir.path.split('/')[-1].split('-')[5][21:] == str(mu_dropdown_value) and 
               dir.path.split('/')[-1].split('-')[8][8:] == str(hidden_dim_dropdown_value)]
    
    return html.Div(
        list(itertools.chain(*zip(
            [dcc.Graph(figure=plot_differences_to_models_mu(dir)) for dir in mu_dirs if plot_differences_to_models_mu(dir) is not None]
        )))
    )

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server
app.layout = dbc.Container([
    html.H1(children='Interactive Early Stopping Mu Plots🫨', style={'textAlign':'center', 'paddingTop': '20px'}),

    html.H4(children='Select number of mus:', style={'textAlign':'center', 'paddingTop': '20px'}),
    dcc.Dropdown([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, "continuous"], id='select-mu-dropdown'),

    html.H4(children='Select model hidden dim:', style={'textAlign':'center', 'paddingTop': '20px'}),
    dcc.Dropdown([4, 16, 64, 256], id='select-hidden-dim-dropdown'),

    html.H3(children='Standard Settings:', style={'textAlign':'center', 'paddingTop': '20px'}),
    dbc.Table(get_settings_table(), bordered=True),
    html.H3(children='Mu Plots:', style={'textAlign':'center', 'paddingTop': '20px'}), 
    html.Div(id='mu-plot-container')]
)

if __name__ == '__main__':
    app.run(debug=True)