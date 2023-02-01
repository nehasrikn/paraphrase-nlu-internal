import plotly.express as px
import os
import sys
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from annotated_data.annotated_data import para_nlu, para_nlu_pretty_names

from paraphrase_utils import get_syntactic_diversity_score, get_lexical_diversity_score
from utils import PROJECT_ROOT_DIR


def plot_lexical_distance_paranlu():
    fig = go.Figure()
    lex_div_traces = []

    for i, (name, dataset) in enumerate(para_nlu.items()):
        lex_div = []
        for bucket in dataset.values():
            for p in bucket:
                lex_div.append(get_lexical_diversity_score(p))
                
        lex_div_traces.append(lex_div)
        
        trace = go.Histogram(x=lex_div, histnorm='percent', name=para_nlu_pretty_names[name], nbinsx=25)
        fig.add_trace(trace)
        
    fig.update_layout(
        autosize=False,
        width=500,
        height=500,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=0
        ),
        xaxis_title="Lexical Distance",
        yaxis_title="Percent",
        legend_title="Data Source",
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.7
        )
    )
    fig.write_image(os.path.join(PROJECT_ROOT_DIR, 'experiments/data_characterization/figures/lexical_distance_paranlu.pdf'))
