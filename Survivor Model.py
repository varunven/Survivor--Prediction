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

def winners_challenge_wins(df):
    """
    Shows all winners and the percent of total challenge wins
    """
    df = df[df['Won']==1]
    plot = sns.catplot(x='Name', y='TChW%', data=df, kind='bar')
    plt.xlabel('Survivor')
    plt.ylabel('Total Challenge Win Percentage')
    plot.set_xticklabels(rotation=45)
    plt.show()

def unanimous_winners(df):
    df = df[df['JV%']==1]
    print(df['Name'])

def most_ind_immunity_wins(df):
    df = df.nlargest(10, 'InICW')
    df = df.sort_values('InICW', ascending=False)
    print(df[['Name', 'InICW', 'Season']])

def select_k_best(df, k):
    X = df.drop(['Won'], axis=1)
    y = df['Won']
    fs = SelectKBest(score_func=f_regression, k=k)
    fs.fit_transform(X, y)
    cols = fs.get_support(indices=True)
    new_df = df.iloc[:,cols]
    new_df['Name_Index'] = df['Unnamed: 0']
    new_df['Won'] = y
    return new_df

def remove_high_correlation(df):
    correlated_features = set()
    correlation_matrix = df.drop('Won', axis=1).corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    df = df.drop(correlated_features, axis=1)
    return df

def svm(df, df_repeaters=None):
    name_to_index = df.filter(['Name', 'Name_Index'])
    df = df.drop('Name', axis=1)
    df = pd.get_dummies(df)
    df = remove_high_correlation(df)
    df = select_k_best(df, 13)
    df = pd.get_dummies(df)
    X = df.drop('Won', axis=1)
    y = df['Won']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    X_original_test = X_test
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    svm = svms.SVC()
    svm.fit(X_train, y_train)
    pred_svm = svm.predict(X_test)
    print(classification_report(y_test, pred_svm))
    print(confusion_matrix(y_test, pred_svm))
    print(accuracy_score(y_test, pred_svm))
    test_model(name_to_index, X_original_test, X_test, y_test, pred_svm)
    winners_at_war_test(svm, X)

def rfc(df, df_repeaters=None):
    name_to_index = df.filter(['Name', 'Name_Index'])
    df = df.drop('Name', axis=1)
    df = pd.get_dummies(df)
    df = remove_high_correlation(df)
    df = select_k_best(df, 13)
    df = pd.get_dummies(df)
    X = df.drop('Won', axis=1)
    y = df['Won']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    X_original_test = X_test
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    rfc = RandomForestClassifier(n_estimators=500)
    rfc.fit(X_train, y_train)
    pred_rfc = rfc.predict(X_test)
    print('RFC')
    print(classification_report(y_test, pred_rfc))
    print(confusion_matrix(y_test, pred_rfc))
    print(accuracy_score(y_test, pred_rfc))
    test_model(name_to_index, X_original_test, X_test, y_test, pred_rfc)

def mlp(df, df_repeaters=None):
    name_to_index = df.filter(['Name', 'Name_Index'])
    df = df.drop('Name', axis=1)
    df = pd.get_dummies(df)
    df = remove_high_correlation(df)
    df = select_k_best(df, 13)
    df = pd.get_dummies(df)
    X = df.drop('Won', axis=1)
    y = df['Won']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    X_original_test = X_test
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    num_features = len(X.columns)
    network = MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes = (num_features,num_features))
    network.fit(X_train, y_train)
    network_pred = network.predict(X_test)
    print('Neural Network MLP')
    print(classification_report(y_test, network_pred))
    print(confusion_matrix(y_test, network_pred))
    print(accuracy_score(y_test, network_pred))
    test_model(name_to_index, X_original_test, X_test, y_test, network_pred)

def logistic_regression(df):
    name_to_index = df.filter(['Name', 'Name_Index'])
    df = df.drop('Name', axis=1)
    df = pd.get_dummies(df)
    df = remove_high_correlation(df)
    df = select_k_best(df, 13)
    X = df.drop('Won', axis=1)
    y = df['Won']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    X_original_test = X_test
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    print('Logistic Regression')
    print(classification_report(y_test, lr_pred))
    print(confusion_matrix(y_test, lr_pred))
    print(accuracy_score(y_test, lr_pred))
    test_model(name_to_index, X_original_test, X_test, y_test, lr_pred)
    winners_at_war_test(lr, X)

def test_model(name_to_index, X_original_test, X_test, y_test, pred):
    y_test = y_test.tolist()
    pred = pred.tolist()
    for i in range(len(X_test)):
        if (pred[i]==1 | y_test[i]==1):
            print(str(X_original_test.iloc[i]['Name_Index'])+'|'+str(y_test[i])+'|'+str(pred[i]))
            ids = name_to_index.index == X_original_test.iloc[i]['Name_Index']
            print(name_to_index.loc[ids,'Name'])
    
def winners_at_war_test(model, X):
    winners_at_war = pd.read_csv(R'C:\Users\dswhi\.vscode\Survivor\winners-at-war.csv')
    name_to_index = winners_at_war.filter(['Name', 'Name_Index'])
    X_war = winners_at_war.filter(X.columns)
    for col in X.columns:
        if col not in X_war.columns:
            X_war[col]=0
    y_war = winners_at_war['Won']
    pred = model.predict(X_war)
    for i in range(len(X_war)):
        if (pred[i]==1):
            ids = name_to_index.index == X_war.iloc[i]['Name_Index']
            print(name_to_index.loc[ids,'Name'])

def main():
    # highest accuracies right now- svm and standard logistic regression
    survivors_total_info = pd.read_csv(R'C:\Users\dswhi\.vscode\Survivor\cleaned_survivors.csv')
    # winners_challenge_wins(survivors_total_info)
    # most_ind_immunity_wins(survivors_total_info)
    svm(survivors_total_info)
    # rfc(survivors_total_info)
    # mlp(survivors_total_info)
    logistic_regression(survivors_total_info)

if __name__ == "__main__":
    main()