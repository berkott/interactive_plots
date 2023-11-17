import os
import json
import re
import itertools
import pathlib
import numpy as np
# from utils.samplers import NormalMeansSampler
# from utils.baseline_models import BayesOptimalPredictor, PrettyGoodPredictor, OraclePredictor
from tqdm import tqdm

import plotly.graph_objects as go
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

def plot_saved_sigma_data(model_save_path):
    format_data = lambda data: {"x":  [float(key) for key in data.keys()], "y": list(data.values())}
    for model_step in pathlib.Path(model_save_path).rglob("*train_step*"):
        with open(model_step, "r") as f:
            raw_data = json.load(f)
            train_step = int(re.sub("[^0-9]", "", model_step.name.split("_")[-1]))

            frame_data_dict = {}


            frame_data_dict["oracle"] = oracle
            frame_data_dict["james_stein"] = james_stein
            frame_data_dict["bayes"] = bayes
            frame_data_dict["learned"] = format_data(raw_data)
            frame_data_dict["train_step"] = int(re.sub("[^0-9]", "", model_step.name.split("_")[-1]))

            frame_data_dicts.append(frame_data_dict)

    frame_data_dicts = sorted(frame_data_dicts, key=lambda x: x["train_step"])

    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    model_signature = model_save_path.split('/')[-1]
    model_name = model_signature.split('-')[0]
    model_params = {model_arg.split('=')[0]: model_arg.split('=')[1] for model_arg in model_signature.split('-')[1:]}

    fig_dict["layout"]["title"] = {"text": f"Out of Distribution sigma {model_name}<br><sup>{str(model_params)}</sup>"}
    fig_dict["layout"]["xaxis"] = {"title": "sigma"}
    fig_dict["layout"]["yaxis"] = {"range": [0, 1.5], "title": "Normalized Risk"}
    fig_dict["layout"]["height"] = 800
    fig_dict["layout"]["legend"] = {"x": 0.85, "y": 0.95}
    fig_dict["layout"]["font"] = {"size": 18}
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 200, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 200,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Epochs:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 200, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    fig_dict["data"] = [
        {
            "x": frame_data_dicts[0]["oracle"]["x"],
            "y": frame_data_dicts[0]["oracle"]["y"],
            "name": "Oracle",
            "mode": "lines+markers"
        },
        {
            "x": frame_data_dicts[0]["james_stein"]["x"],
            "y": frame_data_dicts[0]["james_stein"]["y"],
            "name": "James Stein",
            "mode": "lines+markers"
        },
        {
            "x": frame_data_dicts[0]["bayes"]["x"],
            "y": frame_data_dicts[0]["bayes"]["y"],
            "name": "Bayes",
            "mode": "lines+markers"
        },
        {
            "x": frame_data_dicts[0]["learned"]["x"],
            "y": frame_data_dicts[0]["learned"]["y"],
            "name": "Learned",
            "mode": "lines+markers"
        },
    ]


    fig_dict["frames"] = [
        {
            "name": frame_data_dict["train_step"], 
            "data": [
            {
                "x": frame_data_dict["oracle"]["x"],
                "y": frame_data_dict["oracle"]["y"],
                "name": "Oracle",
                "mode": "lines+markers"
            },
            {
                "x": frame_data_dict["james_stein"]["x"],
                "y": frame_data_dict["james_stein"]["y"],
                "name": "James Stein",
                "mode": "lines+markers"
            },
            {
                "x": frame_data_dict["bayes"]["x"],
                "y": frame_data_dict["bayes"]["y"],
                "name": "Bayes",
                "mode": "lines+markers"
            },
            {
                "x": frame_data_dict["learned"]["x"],
                "y": frame_data_dict["learned"]["y"],
                "name": "Learned",
                "mode": "lines+markers"
            },
        ]} for frame_data_dict in frame_data_dicts
    ]

    sliders_dict["steps"] = [
        {"args": [
            [frame_data_dict["train_step"]],
            {"frame": {"duration": 200},
            "mode": "immediate",
            "transition": {"duration": 200},
            "redraw": False
            }
        ],
        "label": frame_data_dict["train_step"],
        "method": "animate"}
        for frame_data_dict in frame_data_dicts
    ]

    fig_dict["layout"]["sliders"] = [sliders_dict]

    fig = go.Figure(fig_dict)

    return fig

def plot_differences_to_models_mu(model_save_path):
    x_data = []
    model_bayes_cont = []
    model_bayes_disc = []
    model_js = []
    model_true = []

    try:
        with open(os.path.join(model_save_path, "plot_data.json"), "r") as f:
                plot_data_json = json.load(f)
    except:
        return None

    for epoch in plot_data_json.keys():
        x_data.append(int(epoch))
        model_bayes_disc.append(plot_data_json[epoch]["model_bayes_disc"])
        model_bayes_cont.append(plot_data_json[epoch]["model_bayes_cont"])
        model_js.append(plot_data_json[epoch]["model_js"])
        model_true.append(plot_data_json[epoch]["model_true"])

    x_data_numpy = np.array(x_data)
    model_bayes_cont_numpy = np.array(model_bayes_cont)
    model_bayes_disc_numpy = np.array(model_bayes_disc)
    model_js_numpy = np.array(model_js)
    model_true_numpy = np.array(model_true)

    sort_order = np.argsort(x_data_numpy)
    x_data = x_data_numpy[sort_order]
    model_bayes_cont = model_bayes_cont_numpy[sort_order]
    model_bayes_disc = model_bayes_disc_numpy[sort_order]
    model_js = model_js_numpy[sort_order]
    model_true = model_true_numpy[sort_order]

    fig_dict = {
        "data": [],
        "layout": {}
    }

    model_signature = model_save_path.split('/')[-1]
    model_name = model_signature.split('-')[0]
    model_params = {model_arg.split('=')[0]: model_arg.split('=')[1] for model_arg in model_signature.split('-')[1:]}
    keys_to_show = ("plotting_type", "pretraining_mu_tasks", "hid_dim", "layer", "head")
    model_params = {key: model_params[key] for key in model_params if key in keys_to_show}

    fig_dict["layout"]["title"] = {"text": f"Differences mu {model_name}<br><sup>{str(model_params)}</sup>"}
    fig_dict["layout"]["xaxis"] = {"title": "Epochs"}
    fig_dict["layout"]["yaxis"] = {"range": [0, 2], "title": "Normalized MSE"}
    fig_dict["layout"]["height"] = 1000
    fig_dict["layout"]["legend"] = {"x": 0.8, "y": 0.95}
    fig_dict["layout"]["font"] = {"size": 18}

    fig_dict["data"] = [
        {
            "x": x_data,
            "y": model_bayes_cont,
            "name": "Bayes Cont",
            "mode": "lines+markers"
        },
        {
            "x": x_data,
            "y": model_bayes_disc,
            "name": "Bayes Disc",
            "mode": "lines+markers"
        },
        {
            "x": x_data,
            "y": model_js,
            "name": "James Stein",
            "mode": "lines+markers"
        },
        {
            "x": x_data,
            "y": model_true,
            "name": "True mu",
            "mode": "lines+markers"
        },
    ]

    fig = go.Figure(fig_dict)

    return fig

def plot_differences_to_models_sigma(model_save_path):
    format_data = lambda data: {"x":  [float(key) for key in data.keys()], "y": list(data.values())}

    frame_data_dicts = []
    for model_step in pathlib.Path(model_save_path).rglob("*train_step*"):
        with open(model_step, "r") as f:
            raw_data = json.load(f)
            
            frame_data_dict = {}

            frame_data_dict["oracle"] = oracle
            frame_data_dict["james_stein"] = james_stein
            frame_data_dict["bayes"] = bayes
            frame_data_dict["learned"] = format_data(raw_data)
            frame_data_dict["train_step"] = int(re.sub("[^0-9]", "", model_step.name.split("_")[-1]))

            frame_data_dicts.append(frame_data_dict)

    frame_data_dicts = sorted(frame_data_dicts, key=lambda x: x["train_step"])

    x_data = [frame_data_dict["train_step"] for frame_data_dict in frame_data_dicts]
    calc_mse = lambda model, data: [np.mean(np.square(np.array(list(sample[model].values())) - np.array(list(sample["learned"].values())))) for sample in data]

    fig_dict = {
        "data": [],
        "layout": {}
    }

    model_signature = model_save_path.split('/')[-1]
    model_name = model_signature.split('-')[0]
    model_params = {model_arg.split('=')[0]: model_arg.split('=')[1] for model_arg in model_signature.split('-')[1:]}

    fig_dict["layout"]["title"] = {"text": f"Differences sigma {model_name}<br><sup>{str(model_params)}</sup>"}
    fig_dict["layout"]["xaxis"] = {"title": "Epochs"}
    fig_dict["layout"]["yaxis"] = {"range": [0, 2], "title": "MSE"}
    fig_dict["layout"]["height"] = 1000
    fig_dict["layout"]["legend"] = {"x": 0.8, "y": 0.95}
    fig_dict["layout"]["font"] = {"size": 18}
    fig_dict["data"] = [
        {
            "x": x_data,
            "y": calc_mse("oracle", frame_data_dicts),
            "name": "Oracle MSE",
            "mode": "lines+markers"
        },
        {
            "x": x_data,
            "y": calc_mse("james_stein", frame_data_dicts),
            "name": "James Stein MSE",
            "mode": "lines+markers"
        },
        {
            "x": x_data,
            "y": calc_mse("bayes", frame_data_dicts),
            "name": "Bayes MSE",
            "mode": "lines+markers"
        },
    ]

    fig = go.Figure(fig_dict)

    return fig


def get_settings_table():
    table_header = [
        html.Thead(html.Tr([html.Th("Parameter"), html.Th("Value")]))
    ]

    row1 = html.Tr([html.Td("Mus"), html.Td("See meeting notes 10/23")])

    table_body = [html.Tbody([row1])]

    # row1 = html.Tr([html.Td("Sigmas"), html.Td("0.2, 0.5, 0.8")])
    # row2 = html.Tr([html.Td("Mu"), html.Td("0")])
    # row3 = html.Tr([html.Td("N Positions"), html.Td("1024")])
    # row4 = html.Tr([html.Td("Total Epochs"), html.Td("250,001")])
    # row5 = html.Tr([html.Td("Layer Norm"), html.Td("No gradient updates")])
    # row6 = html.Tr([html.Td("Loss"), html.Td("MSE")])
    # row7 = html.Tr([html.Td("Dropout MLPs"), html.Td("None")])
    # row8 = html.Tr([html.Td("Dropout Attention"), html.Td("None")])

    # table_body = [html.Tbody([row1, row2, row3, row4, row5, row6, row7, row8])]

    return table_header + table_body

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

server = app.server

model_dir = "models"

sigma_dirs = [dir.path for dir in os.scandir(model_dir) if dir.is_dir() and dir.path.split('/')[-1].split('-')[1][14:] == "sigma"]
mu_dirs = [dir.path for dir in os.scandir(model_dir) if dir.is_dir() and dir.path.split('/')[-1].split('-')[1][14:] == "mu"]

app.layout = dbc.Container([
    html.H1(children='Interactive Early Stopping Mu Plots🫨', style={'textAlign':'center', 'paddingTop': '20px'}),
    html.H4(children='Standard Settings:', style={'textAlign':'center', 'paddingTop': '20px'}),
    dbc.Table(get_settings_table(), bordered=True)] + 
    [html.H4(children='Sigma Plots:', style={'textAlign':'center', 'paddingTop': '20px'})] + 
    list(itertools.chain(*zip(
        [dcc.Graph(figure=plot_saved_sigma_data(dir)) for dir in sigma_dirs],
        [dcc.Graph(figure=plot_differences_to_models_sigma(dir)) for dir in sigma_dirs]
    ))) + 
    [html.H4(children='Mu Plots:', style={'textAlign':'center', 'paddingTop': '20px'})] + 
    list(itertools.chain(*zip(
        [dcc.Graph(figure=plot_differences_to_models_mu(dir)) for dir in mu_dirs]
    )))
)

if __name__ == '__main__':
    app.run(debug=True)
