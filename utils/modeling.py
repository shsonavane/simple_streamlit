import re
import pandas as pd
import plotly.express as px


def get_sentiment_data(df, text_col, analyzer):
    # gather sentiment scores for data frame `df`
    df_sentiment = []

    for review in df[text_col]:
        vs = analyzer.polarity_scores(review)
        df_sentiment.append(vs)

    df_sentiment = pd.DataFrame(df_sentiment,
                                index = df.index)
    df_sentiment = pd.concat((df, df_sentiment), axis=1)

    return df_sentiment


def plot_sentiment(df_sentiment, benchmarks):
    df_plot = df_sentiment.melt(id_vars=['name', 'roaster'], 
                            value_vars=['neg', 'neu', 'pos', 'compound'],
                            var_name='sentiment_type', value_name='amount')

    fig = px.strip(df_plot, x='sentiment_type', y='amount', 
                template='simple_white', log_y=True,
                hover_name='name',
                hover_data=['roaster'])

    df_ = benchmarks.loc['mean']

    fig.add_scatter(x=df_.index, y=df_, 
                    mode="markers", marker_size=10, marker_color='darkorange', name='review_average')
    
    return fig


def get_sentence_sentiment(text, analyzer):
    sentences = re.split('[?.!]', text)
    sentences = [s for s in sentences if s != '']

    df_ = pd.DataFrame(sentences, 
                       columns=['text'])
    
    df_sentiment = get_sentiment_data(df_, 
                                    text_col='text', 
                                    analyzer=analyzer)
    
    return df_sentiment
# ------------------------------------------------------
# GroupEstimate class  (Week-9 Exercise)
# ------------------------------------------------------
import numpy as np

class GroupEstimate:
    """
    A simple model that groups categorical data and estimates
    a numeric target using mean or median per group.
    """

    def __init__(self, estimate: str = "mean"):
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be either 'mean' or 'median'")
        self.estimate = estimate
        self.group_values_ = None  # stores aggregated y values

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Group by all columns in X and compute mean/median of y."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        df = X.copy()
        df["__y__"] = y.values

        # choose aggregation
        agg_func = np.mean if self.estimate == "mean" else np.median

        # aggregate
        self.group_values_ = (
            df.groupby(list(X.columns))["__y__"].agg(agg_func).reset_index()
        )
        self.group_values_.rename(columns={"__y__": "estimate"}, inplace=True)

    def predict(self, X_new: pd.DataFrame):
        """Return predicted values for X_new based on learned groups."""
        if self.group_values_ is None:
            raise RuntimeError("Model must be fit before calling predict().")

        if isinstance(X_new, list):
            X_new = pd.DataFrame(X_new, columns=self.group_values_.columns[:-1])
        elif not isinstance(X_new, pd.DataFrame):
            raise TypeError("X_new must be a DataFrame or list of lists.")

        merged = pd.merge(
            X_new, self.group_values_,
            on=list(self.group_values_.columns[:-1]),
            how="left"
        )

        missing = merged["estimate"].isna().sum()
        if missing > 0:
            print(f"Warning: {missing} unseen group(s) encountered; returning NaN.")

        return merged["estimate"].values
