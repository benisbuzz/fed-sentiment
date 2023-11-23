from dash import Dash, html, dcc, callback, Output, Input, dash_table
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Union
import numpy as np

app = Dash(__name__)


big_df = pd.read_parquet("data/results/fine_tune_prompt_2/fine_tune_prompt_2_speeches.parquet")



app.layout = html.Div([
    html.H1(children='Title of Dash App', style={'textAlign':'center'}),
    dcc.Dropdown([f"{date.strftime('%d/%m/%Y')} - {title}" for date, title in zip(big_df.index, big_df.iloc[:,1])], big_df.index[0].strftime("%d/%m/%Y"), id='dropdown-selection')
])

def fetch_df(date):
    return pd.DataFrame({"text": big_df.loc[date, "text"], "label": big_df.loc[date, "label"], "reason": big_df.loc[date, "reason"]})

if __name__ == "__main__":
    app.run(debug=True)