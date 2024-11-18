import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
import warnings


warnings.filterwarnings("ignore")

df = pd.read_csv('matches_dataset_original.csv')
df['winner'].fillna('tie', inplace=True)

scores = []
l = df['home_team'].unique().tolist()
l.sort()

home = st.selectbox('Home Team', l)
away = st.selectbox('Away Team', [x for x in l if x!=home])
data = df[['home_team', 'away_team', 'winner', 'home_score', 'away_score']]
data = data[((data['home_team'] == home) | (data['away_team'] == away)) |
            ((data['home_team'] == away) | (data['away_team'] == home))]
data = data.drop_duplicates()

X = data[['home_team', 'away_team']]
y = data[['winner']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Encode teams and winners
team_encoder = LabelEncoder()
winner_encoder = LabelEncoder()

X_train['home_team'] = team_encoder.fit_transform(X_train['home_team'])
X_test['home_team'] = team_encoder.transform(X_test['home_team'])
X_train['away_team'] = team_encoder.transform(X_train['away_team'])
X_test['away_team'] = team_encoder.transform(X_test['away_team'])

y_train = winner_encoder.fit_transform(y_train.values.ravel())
y_test = winner_encoder.transform(y_test.values.ravel())

# Train RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict

team1_encoded = team_encoder.transform([home])[0]
team2_encoded = team_encoder.transform([away])[0]


if st.button('Predict'):
    y_pred_team = model.predict([[team1_encoded, team2_encoded]])
    prob = model.predict_proba([[team1_encoded, team2_encoded]])


    max_prob_index = prob.argmax()
    max_prob = prob[0][max_prob_index]

    if y_pred_team[0] == 0:
        winner = away
        winning_percentage = prob[0][0]
        losing_percentage = prob[0][1]
        tie_percent = prob[0][2]


    elif y_pred_team[0] == 1:
        winner = home
        winning_percentage = prob[0][1]
        losing_percentage = prob[0][0]
        tie_percent = prob[0][2]

    else:
        winner = 'Match tied'

    new_X = data[['winner']]
    new_y = data[['home_score', 'away_score']]

    X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size=0.33, random_state=42)

    winner_encoder = LabelEncoder()
    X_train['winner'] = winner_encoder.fit_transform(X_train['winner'])
    X_test['winner'] = winner_encoder.transform(X_test['winner'])

    home_model = RandomForestRegressor(random_state=42)
    home_model.fit(X_train, y_train['home_score'])

    score_home = home_model.predict([y_pred_team])
    away_model = RandomForestRegressor(random_state=42)
    away_model.fit(X_train, y_train['away_score'])
    score_away = away_model.predict([y_pred_team])

    score_home = np.array(score_home, dtype=np.int64)
    score_away = np.array(score_away, dtype=np.int64)
    winning_goals = int(np.abs(np.subtract(score_home, score_away))[0])

    st.write('Winner: ',winner)
    st.write("Winning goals: ",winning_goals)
    st.write("Winning %: ",format(winning_percentage * 100, '.2f'))
    st.write("Losing %: ",format(losing_percentage * 100, '.2f'))
    st.write("Tie %: ",format(tie_percent * 100,'.2f'))


