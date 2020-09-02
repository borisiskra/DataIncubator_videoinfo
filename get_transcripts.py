
TOPICS = ['Dinosaur', 'Volcano', 'Planet','Bird','Animal','Baseball','Atom','Shark']
from edurank import TOPICS, get_text, get_transcript, get_clean_transcript
from clean_transcript import clean_transcript
# functions for youtube transcript scrape
from youtube_transcript_api import YouTubeTranscriptApi

import pandas as pd

# extract text from transcript
# transcript contains text, starttime and duration of text
# also strip the "[Music]" and "[Applause]" 
fname = 'videos_pandas.pickle'
try:
    #with open(fname, 'r') as f:
    df = pd.read_pickle(fname)
except Exception as err:
    print(err)
    print('No file found')
    exit()

for row in df.iterrows():
    topic = row[1]['Topic']
    videoID = row[1]['videoID']

    print(topic, videoID)
    if videoID == None:
        continue

    fname='transcripts/'+videoID+'.txt'
    try:
        with open(fname,'r') as f:
            transcript = f.read()


    except:
        get_transcript(videoID)
        clean_transcript(videoID)
        transcript = get_clean_transcript(videoID)
    finally:
        test_fname='test/'+topic+'.txt'
        try:
            with open(test_fname, 'a') as f_test:
                f_test.write(transcript+"\n")
        except Exception as err:
            print(err)

