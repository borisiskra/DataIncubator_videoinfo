
# functions for youtube transcript scraping
from youtube_transcript_api import YouTubeTranscriptApi

# extract text from transcript
# transcript contains text, starttime and duration of text
# also strip the "[Music]" and "[Applause]"


def get_text(transcript):
    if transcript == 'Transcript Unavailable':
        return transcript

    listoftext = []
    EXCLUDE = ['[Music]', '[Applause]', '[MUSIC PLAYING]', '[Laughter]']
    for x in transcript:
        if x['text'] in EXCLUDE:
            continue
        listoftext.append(x['text'])

    if listoftext == []:
        return 'Transcript Unavailable'
    # countMusic = len([x['text'] for x in transcript if x['text']  == '[Music]'])
    mytext = ".\n".join(listoftext)
    # return mytext.lower(), countMusic
    return mytext


# check if transcript was previously saved
# if not, get transcript from  youtube.com
def get_transcript(videoID):
    fname = f'data/videos/{videoID}_transcript.txt'
    try:
        with open(fname, 'r') as f:
            transcript = eval(f.read())
            return get_text(transcript)
    except Exception as err:
        print(err)
        print("Trying from youtube.com....")
    try:
        transcript = YouTubeTranscriptApi.get_transcript(videoID)

    except Exception as err:
        print(err)
        transcript = 'Transcript Unavailable'

    with open(fname, 'w', encoding="utf-8") as f:
        f.write(str(transcript))

    return get_text(transcript)


def get_clean_transcript(videoID):
    fname = f'data/videos/{videoID}_transcript.txt'
    try:
        with open(fname, 'r') as f:
            return eval(f.read())

    except Exception as err:
        print(err)
        print("No clean transcript for:", videoID)
