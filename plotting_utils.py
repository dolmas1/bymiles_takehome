import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calc_clever_bins(x, n_bins = 20, max_quartile = 0.95, min_quartile = 0.05):
    bin_edges = [-np.inf, 0, np.inf]
    quantiles = x.quantile([min_quartile, max_quartile]).values
    bin_edges += list(np.linspace(quantiles[0], quantiles[1], num = n_bins))
    
    # Round the labels to as few decimal places as possible, with greater precision until there are no duplicates
    desired_decimals = 0
    bin_labels = [int(x) if not np.isinf(x) else x for x in bin_edges]
    while pd.Series(bin_labels).duplicated().sum() > 0:
        desired_decimals += 1
        bin_labels = [np.round(x, decimals=desired_decimals) for x in bin_edges]
    
    bin_edges.append(0.000001)
    bin_edges.append(-0.000001)
    bin_labels.append(0.000001)
    bin_labels.append(-0.000001)
    
    bin_edges = sorted(bin_edges)
    bin_labels = sorted(bin_labels)
    bin_labels = bin_labels[1:]
    bin_labels[0] = '< '+str(bin_labels[1])
    bin_labels[-1] = '> '+str(bin_labels[-2])

    return bin_edges, bin_labels

def interpret_feature(feat_df, colname, actuals, preds, model, auto_band = None, num_auto_bands = 20):
    """
    Generate visual summary of a feature's impact on the model.

    Parameters:
    -----------
    feat_df : pandas.DataFrame
        DataFrame containing all feature columns required for model.
    colname : str
        Name of the column (feature) in feat_df to analyze.
    actuals : array-like
        Array-like object (e.g., list or numpy array) containing observed target values for each row of feat_df.
    preds : array-like
        Array-like object (e.g., list or numpy array) containing predicted model values for each row of feat_df.
    model : object
        Machine learning model object with a `predict` method capable of predicting on feat_df.

    Returns:
    --------
    pandas.DataFrame
        DataFrame summarizing the average actuals, predicted values, counts, and PDP values per feature level.
        Columns include 'feature' (levels of the feature), 'actual' (average actual target value per level),
        'predicted' (average predicted value per level), 'count' (number of instances per level), and 'PDP'
        (Partial Dependence Plot value per level).

    Notes:
    ------
    This function supports both categorical and numerical features. For categorical features, it calculates
    the average actuals, predicted values, and PDP values per category level. For numerical features, it bins
    the values into discrete levels and performs the same calculations.

    Example usage:
    --------------
    interpret_feature(df, 'FeatureName', y_actual, y_pred, trained_model)
    """

    # Figure out whether we need to band up the feature column
    if auto_band is None:
        if feat_df[colname].dtype == 'object':
            auto_band = False
        elif len(feat_df[colname].unique()) < num_auto_bands:
            auto_band = False
        else:
            auto_band = True
    
    # Categorical Features, and numeric features that don't need banding:
    if not auto_band:
        feat_levels = list(feat_df[colname].unique())

        # Calculate average model prediction, and target value, per level
        df_plot = pd.DataFrame({'feature': feat_df[colname],
                                'actuals': actuals,
                                'preds': preds})
        df_plot = df_plot.groupby('feature').agg(actual=('actuals', 'mean'), predicted=('preds', 'mean'), count=('actuals', 'size'))
        df_plot = df_plot.reset_index()
        
        # Calculate PDP value per level
        partial_deps = []
        for v in feat_levels:
            partial_dep_df = feat_df.copy()
            partial_dep_df[colname] = v
            partial_deps.append(model.predict(partial_dep_df).mean())
        
        # Join PDP values to our summary
        df_plot = pd.merge(df_plot,
                           pd.DataFrame({'feature': feat_levels,
                                         'PDP': partial_deps})
                           )

    # numeric features that need automatically banding:
    else:
        bin_edges, bin_labels = calc_clever_bins(feat_df[colname],
                                                 n_bins = num_auto_bands,
                                                 max_quartile = 0.95,
                                                 min_quartile = 0.05)


        banded_feat = pd.cut(feat_df[colname], bins=bin_edges, labels=bin_labels, right=True, include_lowest=True, retbins=True)[0]

        # Calculate average model prediction, and target value, per level
        df_plot = pd.DataFrame({'feature': banded_feat,
                                'feature_raw': feat_df[colname],
                                'actuals': actuals,
                                'preds': preds})
        df_plot = df_plot.groupby('feature', observed = True).agg(actual=('actuals', 'mean'),
                                                 predicted=('preds', 'mean'),
                                                 avg_feat_val=('feature_raw', 'mean'),
                                                 count=('actuals', 'size'))
        df_plot = df_plot.reset_index()

        
        # Calculate PDP value per level
        partial_deps = []
        for v in df_plot['avg_feat_val']:
            partial_dep_df = feat_df.copy()
            partial_dep_df[colname] = v
            partial_deps.append(model.predict(partial_dep_df).mean())
        
        # Join PDP values to our summary
        df_plot = pd.merge(df_plot,
                           pd.DataFrame({'feature': df_plot['feature'],
                                         'PDP': partial_deps})
                           )

    # Format labels as string to avoid x-axis misalignment issues
    df_plot['feature'] = df_plot['feature'].astype(str)

    # Create combo chart
    fig, ax1 = plt.subplots(figsize=(16, 10))
    
    # bar plot creation
    ax1.set_title(colname, fontsize=16)
    ax1.set_xlabel(colname, fontsize=16)
    ax1.set_ylabel('Quote Count', fontsize=16)
    ax1 = sns.barplot(x='feature', y='count', data=df_plot,
                      color='yellow', saturation=0.8, edgecolor='black')
    ax1.tick_params(axis='y')
    
    # specify we want to share the same x-axis
    ax2 = ax1.twinx()
    
    # line plot creation
    sns.lineplot(x='feature', y='actual', data=df_plot, sort=False,
                 color='fuchsia', linewidth=2,
                 marker='s', markersize=7, markeredgecolor='purple',
                 label='Observed_Avg',
                 ax=ax2)
    sns.lineplot(x='feature', y='predicted', data=df_plot, sort=False,
                 color='darkgreen',  linewidth=2,
                 marker='^', markersize=7, markeredgecolor='darkslategrey',
                 label='Avg_Pred',
                 ax=ax2)
    sns.lineplot(x='feature', y='PDP', data=df_plot, sort=False,
                color='lime', linewidth=2,
                marker='o', markersize=7, markeredgecolor='green',
                label='Model PDP',
                ax=ax2)
    ax2.set_ylabel('Responses', fontsize=16)
    ax2.tick_params(axis='y')
    
    plt.legend()
    plt.show()

    return df_plot