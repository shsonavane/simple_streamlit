import os
import pickle

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

from utils.b2 import B2
from utils.modeling import *

# ------------------------------------------------------
#                DEFINE TRAINING FUNCTIONS
# ------------------------------------------------------

# N/A


if __name__ == '__main__':
    # ------------------------------------------------------
    #                       LOAD DATA
    # ------------------------------------------------------
    # REMOTE_DATA = 'coffee_analysis.csv'

    # load_dotenv()

    # # load Backblaze connection
    # b2 = B2(endpoint=os.environ['B2_ENDPOINT'],
    #         key_id=os.environ['B2_KEYID'],
    #         secret_key=os.environ['B2_APPKEY'])

    # b2.set_bucket(os.environ['B2_BUCKETNAME'])

    # # df_coffee = pd.read_csv('./data/coffee_analysis.csv')
    # df_coffee = b2.get_df(REMOTE_DATA)

    # df_coffee.drop_duplicates(subset='desc_1', 
    #                           inplace=True)
    # df_coffee.dropna(subset=['desc_1', 
    #                          'roast', 
    #                          'loc_country'], 
    #                  inplace=True)

    # ------------------------------------------------------
    #                    TRAIN/SAVE MODEL
    # ------------------------------------------------------

    # Note: this is a simple model which doesn't require the data

    # save this model
    analyzer = SentimentIntensityAnalyzer()

    with open('./model.pickle', 'wb') as f:
        pickle.dump(analyzer, f)