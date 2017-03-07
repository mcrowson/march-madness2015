# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:33:15 2014

@author: Matthew
"""

'''
So we have 4 functions on on here. 
-function getting historic tourney results
-function calculating the logloss of our prediction
-function calculating the tureskill of a team based upon in 
season performnace
-function calulcaitng the probability of team 1 beating team 2.

'''
#Input File Locations
tourneyresults = 'tourney_compact_results.csv'
teamscsv = 'teams.csv'
regseasonresults = 'regular_season_compact_results.csv'
samplesubmission = 'sample_submission.csv'

def get_tourney_results():
    '''
    Our predictions are in a format of season_team1_team2
    So we need to format historic results in the same way.
    Team 1 is the team with a smaller TeamID. We will only return
    the scoring tourney results. Those games played after day 135.
    '''    
    import pandas as pd    
    results = pd.read_csv(tourneyresults)
    results['team1'] = results[['wteam', 'lteam']].min(axis=1)
    results['team2'] = results[['wteam', 'lteam']].max(axis=1)
    results['win'] = 0
    results.ix[results['team1'] == results['wteam'],'win']=1
    results['id'] = results.apply(lambda x: '%s_%s_%s' % (x['season'], x['team1'], x['team2']), axis=1)
    results = results[results['daynum'] > 135]
    y_df = results[['season', 'id', 'win']]
    return y_df

def logloss(y, yhat):
    #Y and Yhat must be vectors of equal length    
    if len(y) != len(yhat):
        raise UserWarning('Y and Yhat are not the same size')

    #We do not predict 0 or 1 as they would make our answers wrong
    not_allowed = [0, 1]
    if len([i for i in yhat if i in not_allowed])>0:
        raise UserWarning('You cannot predict 0 or 1')

    from math import log    
    score = -sum(map(lambda y, yhat: y*log(yhat) + (1-y)*log(1-yhat), y, yhat))/len(y)
    return score


def make_name(season, idx):
    return str(season) + '_' + str(idx)

def true_skill(teams, a_season):
    #Take a season of teams and score them
    #Reads the time file every time
    
    #Initiate a rating object at the beginning of each season and set the score
    #to 0.

    teams['id_s'] = [str(s) + '_' + str(team_id) for team_id in list(teams['team_id'])]
    team_dict = {team_id: ts.Rating() for team_id in teams['team_id']}

    #Play each game and update the ratings
    for idx, row in a_season.T.iteritems():
        winner = team_dict[row['wteam']]
        loser = team_dict[row['lteam']]

        new_w, new_l = ts.rate_1vs1(winner, loser)

        team_dict[row['wteam']] = new_w
        team_dict[row['lteam']] = new_l
        
    #Calculate a score for each team.
    rating_series = pd.Series(team_dict, name='Rating')
    teams = pd.concat([teams.set_index(['team_id']), rating_series], axis=1, join='inner')
    scores = {}
    for idx, row in teams.T.iteritems():
        rating_obj = row['Rating']
        score = rating_obj.mu - 3*rating_obj.sigma
        scores[idx] = score


    teams['Score'] = pd.Series(scores)
    #Return the team data frame with rating object and score values

    return teams[teams['Score'] > 0]

def winprob(matchups, score_seasons):

    from math import copysign    
    #format the season-team combo in our score DF to match.
    #score_seasons['id_s'] = map(lambda s, idx: str(s) + '_' + str(idx), score_seasons['id_s'])
    matchups['pred'] = 0.0

    #It looks like we need to keep the rating objects separate so that
    #we can check the quality of different matches
    for idx, row in matchups.T.iteritems():
        t1_score = score_seasons['Score'][score_seasons['id_s'] == row['t1_s']].item()
        t1_ratingobj = score_seasons['Rating'][score_seasons['id_s'] == row['t1_s']].item()
        t2_score = score_seasons['Score'][score_seasons['id_s'] == row['t2_s']].item()
        t2_ratingobj = score_seasons['Rating'][score_seasons['id_s'] == row['t2_s']].item()           
        qual = ts.quality_1vs1(t1_ratingobj,t2_ratingobj)
        sign = copysign(1, t1_score - t2_score)
        pred = 0.5 + sign*((1 - qual)/2)
        #matchups['pred'][idx] = pred
        matchups.loc[idx, 'pred'] = pred

    #return list of preds equal to matchups len
    
    matchups = matchups.drop('t1_s',1).drop('t2_s',1)
    return matchups
    
def improve_through_tourney(results, teams):
        
    '''
    Using score_seasons as the teams variable here because it has
    the results from multiple seasons combined into 1. 
    '''
    for idx, row in results.T.iteritems():

        if row['pred'] >= .5:
            w = row['t1_s']
            l = row['t2_s']
        else:
            w = row['t2_s']
            l = row['t1_s']
        
        winner = teams['Rating'][teams['id_s'] == w].item()
        loser = teams['Rating'][teams['id_s'] == l].item()
        
        new_w, new_l = ts.rate_1vs1(winner, loser)
        
        teams['Rating'][teams['id_s'] == w] = new_w
        #teams['Rating'][teams['id_s'] == row['lteam']] = new_l
        
    #Calculate a score for each team.
    
    scores = []
    for idx, row in teams.T.iteritems():
        rating_obj = row['Rating']
        score = rating_obj.mu - 3*rating_obj.sigma
        scores.append(score)
    
    teams['Score'] = pd.Series(scores)
    return teams


if __name__ == '__main__':
    #Get historic results
    import time  
    import pandas as pd
    import trueskill as ts
    
    ts.setup()
    start = time.clock()    
    seasons = get_tourney_results()
    last5 = range(2011, 2015)  # ['2011', '2012', '2013', '2014']

    seasons = seasons[seasons.season.isin(last5)]    
    
    season_list = seasons['season'].unique()
    teams = pd.read_csv(teamscsv) 
    reg_season = pd.read_csv(regseasonresults)
    #Get the matchups I need to score
    matchups = pd.read_csv(samplesubmission) 
    matchups = matchups.drop('pred',1)
    #Reduce the matchups to only those seasons for which we are using.
    #matchups = matchups[matchups['id'].isin(seasons['id'])]
    check1 = time.clock()
    
    #print 'Time taken to import and format data ', check1-start    
    
    #All of the seasons are grouped and scored.
    #In the future I could separate all of the seasons and score them sepeartely
    season_teams = []
    for s in season_list:
        a_season = reg_season[reg_season['season'] == s]
        season_teams.append(true_skill(teams, a_season))

    score_seasons = pd.concat(season_teams)
    check2 = time.clock()
    
    #print 'Time taken to run ',len(season_teams),' seasons through ratings.', check2-check1

    
    #Calculate the predictions
    preds = winprob(matchups, score_seasons)

    check3 = time.clock()
    
    #print 'Time taken to calculate post-seasons', check3-check2    
    
    #Limit the predictions to the games actually played starting in day 136
    results = pd.merge(seasons, preds, how='left', on='id')

    '''
    #Improve team ability based upon predicted wins post-season
    post_season_factored = improve_through_tourney(results,score_seasons)  
    new_preds = winprob(matchups,post_season_factored)
    results = pd.merge(seasons,new_preds, how='left', on='id')    
    '''
    
    #Score the predictions    
    y = results['win']
    yhat = results['pred']

    print 'LogLoss score of ',logloss(y, yhat)
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = 'TrueSkill_predictions_' + timestr + '.csv'
    #Lookup how to write a dataframe to file without the index.    
    preds.to_csv(filename)
    print 'file saved as ',filename
