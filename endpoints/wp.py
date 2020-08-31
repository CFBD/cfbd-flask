from flask import Blueprint, request, jsonify
from fastai.tabular import *
import json
import pandas as pd

wp_api = Blueprint('wp_api', __name__)

learner = load_learner('api/models/', 'wp.pkl')
features = ['home_ball', 'adjusted_margin', 'offense_timeouts', 'defense_timeouts', 'down', 'distance', 'yards_to_goal', 'period', 'half_seconds_remaining', 'game_seconds_remaining']

ot_learner = load_learner('api/models/', 'wp_ot.pkl')
ot_features = ['offensive_spread', 'margin', 'down', 'distance', 'yards_to_goal', 'period', 'home_ball', 'is_first']

def getWP(x):
    return learner.predict(x)[2][1].item()

def getWPOT(x):
    return ot_learner.predict(x)[2][1].item()

@wp_api.route("/wp", methods=['POST'])
def wp():
    df = pd.DataFrame.from_records(json.loads(request.data))
    df['o_win_prob'] = df[features].apply(getWP, axis=1)
    df['win_prob'] = np.where(df['home'] == df['offense'], df['o_win_prob'], 1 - df['o_win_prob'])
    ret = df.to_json(orient='table')
    return str(str(ret))


@wp_api.route("/wp/ot", methods=['POST'])
def wp_ot():
    df = pd.DataFrame.from_records(json.loads(request.data))
    
    def is_first(x):
        if df.query("id == {0} and period == {1}".format(x.id, x.period)).sort_values(['drive_number', 'play_number']).iloc[0].offense == x.offense:
            return 1
        else:
            return 0
            
    df['is_first'] = df.apply(is_first, axis=1)

    for feature in ot_features:
        df[feature] = df[feature].astype(float)

    df['o_win_prob'] = df[ot_features].apply(getWPOT, axis=1)
    df['win_prob'] = np.where(df['home'] == df['offense'], df['o_win_prob'], 1 - df['o_win_prob'])
    ret = df.to_json(orient='table')
    return str(str(ret))

@wp_api.route("/test", methods=['GET'])
def test():
    return 'success!'