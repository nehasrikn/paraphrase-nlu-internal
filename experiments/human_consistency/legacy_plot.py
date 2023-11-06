"""
LEGACY PLOTTING CODE FROM THE FIRST SUBMISSION
"""


import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr



def plot_orig_v_bucket_conf(df, plot_title):
    fig = px.scatter(
        df, 
        x="original_confidence", 
        y="conf_shift", 
        #title=plot_title,
        trendline="ols",
        trendline_color_override="blue",
        color='bucket_consistency',
        width=600, 
        height=400,
        color_continuous_scale='Burg',
        hover_data=['example_id', 'bucket_confidence_std'],
        labels={
         "original_confidence": "Model Confidence: Original Example",
         "conf_shift": "Conf Shift: Original ➔ Bucket Mean",
         "bucket_consistency": "consistency",
        }
    )

    fig.update_layout(
        plot_bgcolor='white'
    )
    fig.update_xaxes(
        mirror=True,
        showline=True,
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        showline=True,
        gridcolor='lightgrey'
    )

    
    max_size=5
    size_col = df["bucket_confidence_std"]*2

    sizeref = size_col.max() / max_size ** 2

    fig.update_traces(
        marker=dict(
            sizemode="diameter",
            sizeref=sizeref,
            sizemin=5,
            size=list(size_col),
        ), 
        selector=dict(type='scatter')
    )

    fig.add_trace(go.Scatter(x=[0,1], y=[1,0], mode='lines', name=None, line=dict(color='green', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,-1], mode='lines', name=None, line=dict(color='green', width=1, dash='dot')))
    fig.add_hline(y=0, line_dash="dash")

    fig.update(layout_showlegend=False)
    fig.update_layout(legend=dict(orientation="h"))
    fig.update_xaxes(range=[-0.025, 1.025])
    fig.update_yaxes(range=[-1.1, 1.1])

    fig.update_layout(
        coloraxis_colorbar_orientation = 'h', 
        coloraxis_colorbar_len = 0.2,
        coloraxis_colorbar_thickness=10,
        coloraxis_colorbar_x=0.2,
        coloraxis_colorbar_y=0.05,
        coloraxis_colorbar_title_side='top',
    )


    stat, pvalue = pearsonr(df['original_confidence'], df['conf_shift'])
    a = px.get_trendline_results(fig).px_fit_results.iloc[0].rsquared
    
    fig.add_annotation(x=0.5, y=1.1,
            text=plot_title,
            showarrow=False,
            arrowhead=0,
            font=dict(
                #family="Inconsolata, monospace",
                size=15,
                #color="#8a435d"
            ),
    )

    fig.add_annotation(x=0.9, y=0.8,
            text="Pearson r=%0.2f" % (stat),
            showarrow=False,
            arrowhead=0)
    fig.add_annotation(x=0.9, y=0.7,
            text="pvalue=%0.2f" % (pvalue),
            showarrow=False,
            arrowhead=0)
    fig.add_annotation(x=0.9, y=0.6,
            text="R²=%0.2f" % (a),
            showarrow=False,
            arrowhead=0)
    #fig.show()
    return fig
    
def plot_consistency_cdf(df):
    """
    Empirical Cumulative Distribution Function (ECDF) plot:
    rows of `data_frame` are sorted by the value `x`
    and their cumulative count is drawn as a line.
    """
    fig = px.ecdf(
        df, 
        x="bucket_consistency", 
        markers=False,
        color='model_name',
        ecdfmode="reversed",
        title=plot_title,
        labels={
         "bucket_consistency": "Consistency (% of Bucket)",
         'model_name': 'Model',
         "bilstm": "BiLSTM",

        }
        
    )
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
            xaxis_title="Bucket Consistency (C)",
            yaxis_title="\\% of Buckets with >= x Consistency",
            legend_title="Data Source",
            legend=dict(
                yanchor="top",
                y=0.5,
                xanchor="left",
                x=0.05,
                bgcolor = '#f1f0f5'
            )
    )

    fig.add_annotation(x=0.5, y=1.1,
            text=plot_title,
            showarrow=False,
            arrowhead=0,
            font=dict(
                #family="Inconsolata, monospace",
                size=18,
                #color="#8a435d"
            ),
    )
    fig.update_layout(
        plot_bgcolor='white'
    )
    fig.update_xaxes(
        mirror=True,
        showline=True,
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        showline=True,
        gridcolor='lightgrey'
    )
    

    return fig


def plot_buckets(name: str, bucket_preds: Dict[str, List[str]]):
    metadata = construct_bucket_metadata(bucket_preds)
    plot = plot_orig_v_bucket_conf(metadata, name)
    return plot

def plot_models_acc_v_consistency(model_buckets, test_buckets, dataset):
    data = []

    for model, bucket_preds in model_buckets.items():
        metadata = construct_bucket_metadata(bucket_preds, model_name=model, use_modeling_label=('gpt3' in model))
        data.append({
            'accuracy': accuracy_score(metadata.gold_label, metadata.original_prediction),
            'consistency': np.mean(metadata.bucket_consistency),
            'name': model_pretty_names[model]
        })

    fig = px.scatter(pd.DataFrame(data), x='accuracy', y='consistency', text='name', color='name')
    fig.update_traces(textposition='top center')
    fig.update(layout_showlegend=False)

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
        xaxis_title="Accuracy on ParaNLU Original Examples",
        yaxis_title="Mean Bucket Consistency",
        legend_title="Data Source",
        legend=dict(
            yanchor="top",
            y=0.5,
            xanchor="left",
            x=0.05,
            bgcolor = '#f1f0f5'
        )
    )
    fig.update_layout(yaxis_range=[0,1], xaxis_range=[0,1])

    fig.add_annotation(x=0.6, y=0.97,
        text=para_nlu_pretty_names[dataset],
        showarrow=False,
        arrowhead=0,
        font=dict(
            #family="Inconsolata, monospace",
            size=18,
            #color="#8a435d"
        ),
    )
    # fig.update_layout(
    #     plot_bgcolor='white'
    # )
    # fig.update_xaxes(
    #     mirror=True,
    #     showline=True,
    #     gridcolor='lightgrey'
    # )
    # fig.update_yaxes(
    #     mirror=True,
    #     showline=True,
    #     gridcolor='lightgrey'
    # )




    return fig


def plot_cdf(model_buckets, plot_title):
    model_dfs = []

    for name, bucket_preds in model_buckets.items():
        modeling_label = 'gpt3' in name
        metadata = construct_bucket_metadata(bucket_preds, use_modeling_label=modeling_label, model_name=name)
        model_dfs.append(metadata)

    plot = plot_consistency_cdf(pd.concat(model_dfs), plot_title=plot_title)
    return plot, pd.concat(model_dfs)