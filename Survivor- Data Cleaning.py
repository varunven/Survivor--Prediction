import pandas as pd
import numpy as np 
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn import svm as svms
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from itertools import groupby
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None
"""
Explanation of column names: *** means idk something
Name- first and last
ChW- challenges won both individual and tribal
TotCh- total challenges taken place in
ChW%- ChW/TotCh
SO- number of challenges sat out of
VFB- number of times they voted for person who got out
VAP- votes against player
TotV- total number of votes in the time the player was in the game
TCA- total number of tribal councils attended
TC%- [VFB - (VAP/TotV)] / TCA. Weighted measure of success at tribal council
wTCR- 2* [VFB / (4+VAP)] x (14/TCA). Similar to TC%, but has outside value too
VFT- NO IDEA ***
JVF- jury votes for player
TotJ- total jury number in the season
JV%- percentage of jury votes for player
SurvSc- ChW% + TC% + JV%. Maximum possible score of 2 (3 after the finale)
SurvAv- ChW + wTCR + JV%. Maximum 18
Days- total days played per season
Place- final placing
Non-VFB- total number of votes cast for the person who did not get out
InRCA- individual reward challenge attempts
InRCW- individual reward challenge wins
InICA- individual immunity challenge attempts
InICW- individual immunity challenge wins
InChA- individual total challenge attempts
InChW- individual total challenge wins
Season- season number
TRCA- team reward challenge attempts
TRCW- team reward challenge wins
TICA- team immunity challenge attempts
TICW- team immunity challenge wins
TChA- team challenge attempts
TChW- team challenge wins
TChW%- team challenge win percentage
Exile- number of days spent on exile island. All players are 0 if exile island isn't a thing
features to consider adding- 
% of tribe in original alliance- lots of work
% of votes received pre merge- lots of work
% of tribe going into the merge- lots of work
% of votes out of loop on OR % of votes known about- lots of work
education Level- where?
population of city- TONS OF WORK
marital status- hard to fill in
if in majority of first vote- lots of work
blamed for loss at a challenge (at least once)- lots of work
laziness score- prob 0 or 1 (1 being lazy)
did they have a person they trusted in completely- lots of work
leader of tribe before the merge- lots of work
if tribe mate complained about their personality- lots of work
overheard information that others did not know they heard
"""

def initial_cleaning(survivors):
    #edit if it contains #DIV/0
    survivors = survivors.replace({'#DIV/0!': 0})
    #retain everything but VFT (dont know what that means) and jv% and jvf because those should not be known
    #drop place (duh) and replace survav and survsc bc those use jv stats
    survivors['SurvSc'] = survivors['SurvSc']-survivors['JV%'] 
    survivors['SurvAv'] = survivors['SurvAv'].astype(float)
    survivors['SurvAv'] = survivors['SurvAv']-6*survivors['JV%'] 
    survivors = survivors.drop(['JV%', 'JVF','VFT', 'Place'], axis=1)
    #turn strings into numeric columns
    survivors['TChW%'] = pd.to_numeric(survivors['TChW%'])
    survivors['wTCR'] = pd.to_numeric(survivors['wTCR'])
    #retain age, sex, state, and winner of the season
    survivors_age = pd.read_csv(R'C:\Users\dswhi\.vscode\Survivor\survivors.csv')
    survivors_age = survivors_age.filter(['name', 'age', 'sex', 'season.num', 'state', 'winner'])
    survivors_total_info = survivors.merge(survivors_age, left_on = ['Name', 'Season'], right_on=['name', 'season.num'])
    survivors_total_info = survivors_total_info.drop(['name', 'season.num'], axis=1)
    #make a column showing if they won their season or not
    survivors_total_info['Won'] = 0
    is_winner = survivors_total_info['Name'].eq(survivors_total_info['winner'])
    survivors_total_info.loc[is_winner, 'Won']+=1
    survivors_total_info = survivors_total_info.drop('winner',  axis=1)
    grouped = [list(g) for k, g in groupby(survivors_total_info['Name'].tolist())]
    survivors_total_info['Name_Index'] = np.repeat(range(len(grouped)),[len(x) for x in grouped])
    survivors_total_info['state_D.C.'] = 0
    return survivors_total_info

def feature_addition(survivors):
    github_survivors = pd.read_csv(R'C:\Users\dswhi\.vscode\Survivor\github-survivor-data.csv')
    github_survivors = github_survivors.drop(['age', 'hometown', 'finish', 'tribalChallengeWins',
    'individualChallengeWins', 'totalWins', 'daysLasted', 'votesAgainst'], axis=1)
    github_survivors = github_survivors[github_survivors.columns.drop(list(github_survivors.filter(regex='PictureURL')))]
    github_survivors = github_survivors[github_survivors.columns.drop(list(github_survivors.filter(regex='trivia')))]
    merged = survivors.merge(github_survivors, left_on=['Name', 'Season'], right_on=['contestant', 'season'])
    idols = pd.read_csv(R'C:\Users\dswhi\.vscode\Survivor\idols.csv')
    idols = idols.fillna(0)
    idols['Season'] = idols['Season'].str.strip('S')
    idols['Season'] = pd.to_numeric(idols['Season'])
    idols['Contestant'] = idols['Contestant'].str.replace('\d+', '')
    idols['Contestant'] = idols['Contestant'].str.replace('-', '')
    idols = idols.groupby(['Contestant', 'Season']).agg({'Idols found':'sum', 'Idols held':'sum',
    'Idols played':'sum', 'Votes voided':'sum', 'Boot avoided':'max', 'Tie avoided':'max'}).reset_index()
    merged = merged.merge(idols, left_on=['Name', 'Season'], right_on=['Contestant', 'Season'], how='left')
    merged = merged.fillna(0)
    merged = merged.drop(['contestant', 'Contestant', 'sex'], axis=1)
    merged['gender'] = (merged['gender'] == 'Male').astype(int)
    merged = merged.sort_index()
    return merged

def main():
    survivors = pd.read_csv(R'C:\Users\dswhi\.vscode\Survivor\survivors2.csv')
    survivors_total_info = initial_cleaning(survivors)
    survivors_total_info = feature_addition(survivors_total_info)
    survivors_total_info.to_csv(R'C:\Users\dswhi\.vscode\Survivor\cleaned_survivors.csv')

if __name__ == "__main__":
    main()