#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:34:14 2019

@author: mac
"""

import os
import pandas as pd
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow

# The CLIENT_SECRETS_FILE variable specifies the name of a file that contains
# the OAuth 2.0 information for this application, including its client_id and
# client_secret.
CLIENT_SECRETS_FILE = "client_secret.json" #This is the name of your JSON file

# This OAuth 2.0 access scope allows for full read/write access to the
# authenticated user's account and requires requests to use an SSL connection.
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

def get_authenticated_service():
  flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
  credentials = flow.run_console()
  return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
service = get_authenticated_service()

# =============================================================================
# Search Query Initialisation
# =============================================================================

count = 0


def get_songTitles():
    data = pd.read_csv("metadata.csv")
    return data['artist'].tolist(),data['title'].tolist()
artist, song_titles  = get_songTitles() #returns 2 list containing artist and song titles

# To start at the first song after the last song in which comments were collected. Take note of the ending index, change range to begin at index after the last song in which comments were collected. Take note of ending index after each iteration.
for i in range(len(song_titles)):
    if count == 0:
        artist_temp = artist[i]
        song_temp = song_titles[i]
        song_adv = "\""+song_temp+"\""
        query_results = service.search().list(
                part = 'snippet',
                q = artist_temp +' '+ song_temp,
                order = 'relevance', # You can consider using viewCount
                maxResults = 5,
                type = 'video', # Channels might appear in search results
                relevanceLanguage = 'en',
                safeSearch = 'none',
                ).execute()
        for vid in range(len(query_results['items'])):
            query_results['items'][vid]['query'] = str(artist_temp +' '+ song_temp)
            query_results['items'][vid]['queryindex'] = int(vid + 1)
        count = count + 1
    elif count < 3:
        artist_temp = artist[i]
        song_temp = song_titles[i]
        song_adv = "\""+song_temp+"\""
        query_results_temp = service.search().list(
                part = 'snippet',
                q = artist_temp +' '+ song_temp,
                order = 'relevance', # You can consider using viewCount
                maxResults = 5,
                type = 'video', # Channels might appear in search results
                relevanceLanguage = 'en',
                safeSearch = 'none',
                ).execute()
        for vid in range(len(query_results_temp['items'])):
            query_results_temp['items'][vid]['query'] = str(artist_temp +' '+ song_temp)
            query_results_temp['items'][vid]['queryindex'] = int(vid + 1)
        query_results['items'] += query_results_temp['items']
        count = count + 1
    else:
        break




# =============================================================================
# Get Video IDs
# =============================================================================
video_id = []
channel = []
video_title = []
video_desc = []
query = []
query_index = []

for item in query_results['items']:
    video_id.append(item['id']['videoId'])
    channel.append(item['snippet']['channelTitle'])
    video_title.append(item['snippet']['title'])
    video_desc.append(item['snippet']['description'])
    query.append(item['query'])
    query_index.append(item['queryindex'])
    
# =============================================================================
# Get Comments of Top Videos
# =============================================================================
video_id_pop = []
channel_pop = []
video_title_pop = []
video_desc_pop = []
query_pop = []
query_index_pop = []
comments_pop = []
comment_id_pop = []
comment_index_pop = []
reply_count_pop = []
like_count_pop = []

count_2 = 0
from tqdm import tqdm
for i, video in enumerate(tqdm(video_id, ncols = 10)):
    try:
        if count_2 == 0:
            response = service.commentThreads().list(
                    part = 'snippet',
                    videoId = video,
                    maxResults = 10, # Only take top 100 comments...
                    order = 'relevance', #... ranked on relevance
                    textFormat = 'plainText',
                    ).execute()
            for idx in range(len(response['items'])):
                response['items'][idx]['commentindex'] = int(idx + 1)
            count_2 = count_2 + 1
        else:
            response_temp = service.commentThreads().list(
                    part = 'snippet',
                    videoId = video,
                    maxResults = 10, # Only take top 100 comments...
                    order = 'relevance', #... ranked on relevance
                    textFormat = 'plainText',
                    ).execute()
            for idx in range(len(response_temp['items'])):
                response_temp['items'][idx]['commentindex'] = int(idx + 1)
            response['items'] += response_temp['items']
    except:
        next
        
    
    comments_temp = []
    comment_id_temp = []
    comment_index_temp = []
    reply_count_temp = []
    like_count_temp = []
    for idx in range(len(response['items'])):
        print(idx)
        comments_temp.append(response['items'][idx]['snippet']['topLevelComment']['snippet']     ['textDisplay'])
        comment_id_temp.append(response['items'][idx]['snippet']['topLevelComment']['id'])
        comment_index_temp.append(response['items'][idx]['commentindex'])
        reply_count_temp.append(response['items'][idx]['snippet']['totalReplyCount'])
        like_count_temp.append(response['items'][idx]['snippet']['topLevelComment']['snippet']['likeCount'])
    comments_pop.extend(comments_temp)
    comment_id_pop.extend(comment_id_temp)
    comment_index_pop.extend(comment_index_temp)
    reply_count_pop.extend(reply_count_temp)
    like_count_pop.extend(like_count_temp)
        video_id_pop.extend([video_id[i]]*len(comments_temp))
        channel_pop.extend([channel[i]]*len(comments_temp))
        video_title_pop.extend([video_title[i]]*len(comments_temp))
        video_desc_pop.extend([video_desc[i]]*len(comments_temp))
        query_pop.extend([query[i]]*len(comments_temp))
        query_index_pop.extend([query_index[i]]*len(comments_temp))

        


    
# =============================================================================
# Populate to Dataframe
# =============================================================================
import pandas as pd

output_dict = {
        'Query': query_pop,
        'Query Index': query_index_pop,
        'Channel': channel_pop,
        'Video Title': video_title_pop,
        'Video Description': video_desc_pop,
        'Video ID': video_id_pop,
        'Comment': comments_pop,
        'Comment ID': comment_id_pop,
        'Comment Index': comment_index_pop,
        'Replies': reply_count_pop,
        'Likes': like_count_pop,
        }

output_df = pd.DataFrame(output_dict, columns = output_dict.keys())

# Create CSV file of collected comments
import csv

output_df.to_csv('comments.csv', encoding='utf-8', mode='a')
