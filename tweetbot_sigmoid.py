
# coding: utf-8

# In[ ]:




# Performs a user_timeline search every 2 minutes.  If a "#ServiceAlert" appears indicating a delay, then it posts whether it will be longer/shorter than 10 minutes.

# In[1]:

#!/usr/bin/env/python
import tweepy
import time as tyme
import numpy as np
#from sklearn.externals import joblib
import math as mth
import random as rd
import pandas as pd
import csv
import difflib
import datetime as dt

# mtadelaybot@gmail.com keys
consumer_key = 'aqkD8HhCYeJT5rYvF2VXa09dA'
consumer_secret = '0P5KOQQwD1VjfBzJlOB62hReOdETVj2qp8RwT2VDKFlbH2HPSs'
access_key = '873036002226388997-d5TUF6BeMOiOZTP3qpdl172ESLF6dy7'
access_secret = 'thg7RZfonucmdMmsJlj776Sr0zGBAxoZFrlCOhTsD1on8'

CONSUMER_KEY = consumer_key
CONSUMER_SECRET = consumer_secret
ACCESS_TOKEN = access_key
ACCESS_TOKEN_SECRET = access_secret

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.secure = True
auth.set_access_token(ACCESS_TOKEN,ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

# load the prediction model
#filename = 'ridge_model20.sav'
#loaded_model = joblib.load(filename)

# because I can't import sklearn into heroku, I have to calculate the probability
# with the classifier weights and the sigmoid function
def sigmoid(x):
  return 1 / (1 + mth.exp(-x))

# these are the ridge logistic regression classifier weights.  Input the twtvec
# and this will return the classification
def make_prediction(twtvec):
    
    #15 minute classification model
    modelweights = [-0.00662503, -0.17817792, -0.02037231, -0.0210132 ,  0.06548153,
        -0.00790169, -0.01603905,  0.10665057,  0.04730125, -0.06777136,
        -0.08618045,  0.01281334, -0.31174278, -0.41968088, -0.17311834,
         0.3508457 ,  0.31669431,  0.22418866,  0.00413489,  0.05554019,
        -0.02157884, -0.03809624, -0.10488451,  0.01699535, -0.05019072,
         0.05302499,  0.08505489, -0.00448227,  0.00448227]
    modelweights = np.asarray(modelweights)
    tv = np.asarray(twtvec)
    tv = tv.reshape(1,-1)
    s = (np.dot(tv,modelweights))
    
    prob = sigmoid(s)
    if prob<0.5:
        return 0
    else:
        return 1


# feed the tweet info into the feature extractor 
# returns the vectorized information as below:
def clean_tweet_online(twt):
    
    # looks for a substring within each of the sublists
    # and returns the sublist index where it is found
    def find_feature(featurestr,twtveclabels):
        l_ct = -1
        for l in twtveclabels:
            l_ct = l_ct + 1
            wh_idx = l.find(featurestr)
            if wh_idx != -1:
                break
        return l_ct

    # bin the day into weekday or weekend
    def bin_weekday(wkday):
        if wkday in [0,1,2,3,4]:
            return 'weekday'
        else:
            return 'weekend'

    # bin the month into a season
    def bin_month(month):
        win = [12,1,2]
        spr = [3,4,5]
        summ = [6,7,8]
        aut = [9,10,11]
        if month in win:
            return 'winter'
        elif month in spr:
            return 'spring'
        elif month in summ:
            return 'summer'
        else:
            return 'autumn'

    # bin the hour into a time of day
    def bin_time(hour):
        rush_eve = [19,20,21] #GMT
        rush_morn = [10,11,12]
        t_work = [13,14,15,16,17,18]
        t_eve = [22,23,0]
        t_sleep = [1,2,3,4,5,6,7,8,9]

        if hour in rush_eve:
            return 'time_rush_hour_eve'
        elif hour in rush_morn:
            return 'time_rush_hour_morn'
        elif hour in t_work:
            return 'time_work'
        elif hour in t_eve:
            return 'time_eve'
        else:
            return 'time_sleep'

    # find the best match of location indicated in the tweet to the actual station name
    # this will help reduce the size of the joined table in SQL
    def find_best_match(loc,names):
        closest = difflib.get_close_matches(loc,names)  
        if not closest:
            return loc
        else:
            return closest[0]

    # recategorize the 'what_happened' column to be more general
    # this will help reduce the size of the feature set when implementing one-hot-encoding
    def generalize_event(event):
        happened_list = [['signal'],['passenger','customer'],['mechanical'],
                     ['track','rail'],['switch'],
                     ['fdny activity','nypd activity','investigation']]

        happened_replace = ['signal','passenger','mechanical','track','switch','investigation']

        happened_ct = 0
        for j in happened_list:
            for k in j:
                if k in event:
                    return happened_replace[happened_ct]
            happened_ct = happened_ct+1

        return 'reason_undetermined'

    # feature labels
    twtveclabels = ['latitude','longitude','structure_At Grade','structure_Elevated',
                    'structure_Open Cut','structure_Subway','structure_Viaduct','direction_b/d',
                    'direction_direction_unknown ','direction_n/b','direction_s/b',
                    'what_happened_investigation','what_happened_mechanical',
                    'what_happened_passenger','what_happened_reason_undetermined',
                    'what_happened_signal','what_happened_switch','what_happened_track',
                    'winter','spring','summer','autumn','time_rush_hour_eve',
                    'time_rush_hour_morn','time_work','time_eve','time_sleep','weekday',
                    'weekend']

    # load station information
    # station information
    stations = pd.read_csv('Stations.csv')
    #there are multiple coordinates associated with each stop -- for now just take the first
    stations = stations.drop_duplicates(['stop_name'],keep='first')
    names = list(stations['stop_name']) #station names
    lat = list(stations['latitude']) # latitude coordinates
    lon = list(stations['longitude']) # longitude coordinates
    stru = list(stations['structure']) # type of subway station structure

    # calculate min and max coordinates to normalize them
    latmax = np.asarray(lat).max()
    latmin = np.asarray(lat).min()
    lonmax = np.asarray(lon).max()
    lonmin = np.asarray(lon).min()

    # holds feature values
    twtvec = [0] * len(twtveclabels)

    # make sure the information matches up
    subservicealert = str(twt[2])

    # find where the delay start happened
    subat_idx = subservicealert.find(' at ') #index of word 'at'
    subat_str = subservicealert[subat_idx:]
    subp_idx = subat_str.find('.')
    loc = subat_str[4:subp_idx]
    loc = find_best_match(loc,names)

    # find the coordinates at whih it happened
    idx = names.index(loc)
    lati = (lat[idx]-latmin)/latmax # normalized latitude coordinate
    longi = (lon[idx] - lonmin)/lonmax # normalized longitude coordinate

    # insert coordinate location into vector
    fidx = find_feature('latitude', twtveclabels)
    twtvec[fidx] = lati
    fidx = find_feature('longitude', twtveclabels)
    twtvec[fidx] = longi

    # type of subway structure
    structure = stru[idx]

    # instert structure type into vector
    if structure is not 'Embankment':
        fidx = find_feature('structure_' + structure, twtveclabels)
        twtvec[fidx] = 1

    # extract what happened
    if 'due to' in subservicealert:
        due_idx = subservicealert.find(' due to ') #index of words 'due to'
        due_str = subservicealert[due_idx+7:subat_idx]
        what_happened = generalize_event(due_str)
    else:
        what_happened = 'reason_undetermined'

    # insert it in the vector
    fidx = find_feature('what_happened_' + what_happened, twtveclabels)
    twtvec[fidx] = 1

    # determine direction of train    
    if 'b/d' in subservicealert:
        direction = 'b/d'
    elif 'n/b' in subservicealert:
        direction = 'n/b'
    elif 's/b' in subservicealert:
        direction = 's/b'
    else:
        direction = ' '

    # insert it in the vector
    fidx = find_feature('direction_' + direction, twtveclabels)
    twtvec[fidx] = 1

    # time at the start of the delay (needs binning)
    startwhen = twt[1]
    #print(dt.datetime.now())
    startwhen = dt.datetime.strptime(str(startwhen), '%Y-%m-%d %H:%M:%S')
    #startwhenlocal = startwhen.astimezone(timezone('US/Eastern'))
    #timeonly = startwhen.time().strftime('%H:%M:%S')
    timeonly = (startwhen - dt.timedelta(hours=4)).time().strftime('%H:%M:%S')

    # bin the time
    event_time = bin_time(startwhen.hour)
    fidx = find_feature(event_time, twtveclabels)
    twtvec[fidx] = 1

    # bin the day / month
    event_month = bin_month(startwhen.month)
    event_weekday = bin_weekday(startwhen.weekday())

    # insert them into vector
    fidx = find_feature(event_month, twtveclabels)
    twtvec[fidx] = 1
    fidx = find_feature(event_weekday, twtveclabels)
    twtvec[fidx] = 1

    return twtvec,loc,timeonly



# Now periodically check the NYCTSubway twitter feed for new info.
# If a delay is announced, make a prediction
ct =  20 # only pull this many tweets
newest = []
new_tweets = []

while 1!=0:
    
    if not newest: # if newest is empty, there is no previous tweet
        # makes an initial request for most recent tweets
        new_tweets = api.user_timeline('@NYCTSubway', count = ct)
    else:
        new_tweets = api.user_timeline('@NYCTSubway', since_id = newest)

    
    if not new_tweets: # if there are no new tweets (or no tweets at all)
        print('no new tweets')
    else:
        
        outtweets = ([[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] 
                      for tweet in new_tweets])
        
        delayflag = 0
        for j in range(0,len(outtweets)):
            
            # search new tweets for #ServiceAlert term
            if ('Allow additional travel ' in str(outtweets[j][2]) and 
                 'delay' in str(outtweets[j][2])): 
                
                delayflag = delayflag + 1
                twt = outtweets[j]

                # vectorize the tweet
                twtvec,loc,tm = clean_tweet_online(twt)

                #the output will be determined by the trained model (ridge):
                #tv = np.asarray(twtvec)
                #tv = tv.reshape(1,-1)
                #prediction = loaded_model.predict(tv)
                prediction = make_prediction(twtvec)
                if prediction == 1:
                    line = ('The delay at ' + loc + ', beginning at ' + 
                            tm[0:5] + ' is expected to be greater than 15 minutes. ') 
                else:
                    line = ('The delay at ' + loc + ', beginning at ' + 
                            tm[0:5] + ' is expected to be less than 15 minutes. ') 
                            
                # add link to the tweet
                link = 'https://twitter.com/i/web/status/' + str(twt[0])
                line = line + 'See ' + link

                # check that this tweet isn't identical to the last one
                # this might cause a tweepy 403 error
                mytweets = api.user_timeline('@MTADelayBot',count = 1)
                mylasttweet = mytweets[0].text
                
                if line != mylasttweet:
                    # retweet the @NYCTSubway tweet
                    #api.retweet(int(twt[0]))
                    print(twt)
                    print(line)
                    # update the MTADelayBot timeline
                    api.update_status(status = line)
                    #api.me(line)
                    tyme.sleep(rd.randint(20,30))
                else:
                    print('this is a repeat tweet -- let us skip it ')
                
        # save the id of the newest tweet plus one (plus 1 to avoid overlap)
        newest = new_tweets[0].id + 1
        
        print('Newest tweed ID is ' + str(newest))
        
        if delayflag==0:
            print('There are tweets, but none announcing new delays')
        else:
            print('There were ' + str(delayflag) + ' delay announcements')
        
    # sleep for a bit to avoid rate limit errors
    tyme.sleep(rd.randint(240,300))


