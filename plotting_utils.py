import numpy as np
import seaborn as sns
import matplotlib

def plot_isocurves(ax: matplotlib.axes.Axes):
    
    # plot the isocurves
    bernoullis = np.linspace(0, 1, 1000)
    total_variance = bernoullis * (1-bernoullis)
    
    pove_percentages = [i / 10 for i in range(10)]
    palette = sns.color_palette("flare", len(pove_percentages)) #crest
    
    for i, pove in enumerate(pove_percentages):
        min_pstay = 1 - 2*((1-pove)*total_variance) # 2 * UV, but max is when all variance is UV
        ax.plot(bernoullis, min_pstay, '--', alpha=0.3, color=palette[i])
        annotation_point = 500
        curve_text = f"{100 - int(100 * pove)}% PVAP" if pove > 0 else r"Min $P_C$"
        ax.annotate(
            curve_text, 
            (bernoullis[annotation_point], min_pstay[annotation_point]+0.009), 
            rotation=0, 
            horizontalalignment='center', 
            verticalalignment='bottom', 
            alpha=0.3, 
            color='black',
            fontsize=8
        )