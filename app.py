import pickle
import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils.preprocessing import *
from utils.modeling import *


# ------------------------------------------------------
#                      APP CONSTANTS
# ------------------------------------------------------
MODELPATH = './model.pickle'

# ------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------
st.set_page_config(
    page_title="Coffee Sentiment Analyzer",
    page_icon="☕",  # optional, you can remove or change this
    layout="wide"    # makes better use of screen space
)

# ------------------------------------------------------
#                  CACHE DATA AND MODEL
# ------------------------------------------------------
@st.cache_resource
def get_model():
    with open(MODELPATH, 'rb') as f:
        analyzer = pickle.load(f)
    return analyzer


@st.cache_data
def get_benchmarks(df_coffee):
    # average sentiment scores for the whole dataset
    benchmarks = df_coffee[['neg', 'neu', 'pos', 'compound']].agg(['mean', 'median'])
    return benchmarks


# ------------------------------------------------------
#                  LOAD DATA (Final Version)
# ------------------------------------------------------
@st.cache_data(ttl="5m")
def load_coffee_data():
    """Try to load from Google Sheets; fallback to GitHub and compute sentiment."""
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(ttl="5m")
        st.success("Loaded data from Google Sheets")
    except Exception as e:
        st.warning(f"Google Sheets failed ({e}); loading from GitHub instead.")
        df = pd.read_csv(
            "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
        )

        # compute sentiment columns dynamically
        analyzer = SentimentIntensityAnalyzer()
        df = get_sentiment_data(df, text_col="desc_1", analyzer=analyzer)

        st.success("Loaded data from GitHub CSV and computed sentiment columns")

    return df


# ------------------------------------------------------
#                INITIALIZE DATA AND MODEL
# ------------------------------------------------------
df_coffee = load_coffee_data()
analyzer = get_model()
benchmarks = get_benchmarks(df_coffee)

# Optional debug check
st.write("Columns in dataset:", list(df_coffee.columns))


# ------------------------------------------------------
#                         APP
# ------------------------------------------------------
# ------------------------------
# PART 0 : Overview
# ------------------------------
st.write(
    '''
# Review Sentiment Analysis  
We pull data from Google Sheets (or fallback to GitHub), analyze it, and visualize sentiment.
'''
)

# ------------------------------
# PART 1 : Filter Data
# ------------------------------
roast = st.selectbox("Select a roast:", df_coffee['roast'].unique())
loc_country = st.selectbox("Select a roaster location:", df_coffee['loc_country'].unique())

# note that this function caches data returned based on repeated input
df_filtered = filter_coffee(roast, loc_country, df_coffee)

st.write("**Your filtered data:**")
st.dataframe(df_filtered)

# ------------------------------
# PART 2 : Plot
# ------------------------------
st.write(
    '''
## Visualize  
Compare this subset of reviews with the rest of the dataset.
'''
)

fig = plot_sentiment(df_filtered, benchmarks)
st.plotly_chart(fig)

# ------------------------------
# PART 3 : Analyze Input Sentiment
# ------------------------------
st.write(
    '''
## Custom Sentiment Check  
Compare these results with the sentiment scores of your own input.
'''
)

text = st.text_input("Write a paragraph, if you like.", "Your text here.")
df_sentiment = get_sentence_sentiment(text, analyzer)
st.dataframe(df_sentiment)
