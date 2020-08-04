from flask import Blueprint, request, jsonify
from fastai.tabular import *
import json
import pandas as pd

wp_api = Blueprint('wp_api', __name__)

learner = load_learner('api/models/', 'wp.pkl')
features = ['home_ball', 'adjusted_margin', 'offense_timeouts', 'defense_timeouts', 'down', 'distance', 'yards_to_goal', 'period', 'half_seconds_remaining', 'game_seconds_remaining']

def getWP(x):
    return learner.predict(x)[2][1].item()

@wp_api.route("/wp", methods=['POST'])
def wp():
    df = pd.DataFrame.from_records(json.loads(request.data))
    df['o_win_prob'] = df[features].apply(getWP, axis=1)
    df['win_prob'] = np.where(df['home'] == df['offense'], 1 - df['o_win_prob'], df['o_win_prob'])
    ret = df.to_json(orient='table')
    return str(str(ret))