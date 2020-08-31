import time

import numpy as np

import json

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


from flask import Flask, request, render_template

from bokeh.embed import json_item
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import CDN

from bokeh.models import ColumnDataSource#, Plot, LinearAxis, Grid, LabelSet
from bokeh.models.glyphs import HBar, Text

# local imports
from bert_classify import bert_video_classifier

app = Flask(__name__)


@app.route("/") # 
def index(search=None, num_videos = 0, **kwargs):
    print('/')
    return render_template("index.html", 
                           resources=CDN.render(), 
                           num_videos = num_videos, 
                           **kwargs
                          )


@app.route('/', methods=['POST'])
def index_post():
    
    text = request.form['search']
    topic = text.capitalize()
    print(topic)
    videos_lst = get_ids_from_youtube(topic, 10)
    
    #videos_lst = [('noiwY7kQ5NQ', 'Title')]#get_ids_from_youtube(topic)
    
    print(videos_lst)
    topics_histogram_lst = []
    words_histogram_lst = []
    
    for videoID, _ in videos_lst:
        topics_counter, words_counter = bert_video_classifier(videoID)
        topics_histogram_lst.append(
            make_plot(topics_counter, top_n=3, percent=True)
            )
        words_histogram_lst.append(
            make_plot(words_counter, top_n=5, percent=False)
            )


    return index(search=topic,
                 num_videos = len(videos_lst),
                 videos_lst = videos_lst,
                 topics_histogram_lst = topics_histogram_lst,
                 words_histogram_lst = words_histogram_lst
                 )


def make_plot(counter, top_n=5, percent=False ):
    
    
    most_common = counter.most_common(top_n)
    print(most_common)
    most_common.sort(key=lambda x: x[1])
    print(most_common)
    
    words, count = list( zip( *most_common) )
    labels = list(map(lambda x: str(x),count))
    
    print(words)
    print(count)
    x_pos = list(map(lambda x: max(2,x-1), count))
    
    y_pos = np.arange(len(words)) + .5
    if percent:
        total = sum(count)
        count = np.round(100*np.array(count)/total,1)
        x_pos = list(map(lambda x: max(2,x-20), count))
        labels = list(map(lambda x: str(x)+"%",count))
        
    print(count)
    
    
    source = ColumnDataSource(dict(x=x_pos,
                                   y=y_pos,
                                   labels=labels))

    plot = figure(plot_width=300, plot_height=150, y_range=words)
    plot.hbar(y=y_pos, right=count, left=0, height=0.5, color="#FF9E00")

    plot.toolbar.logo = None
    plot.toolbar_location = None
    plot.yaxis.major_label_text_font_size = "15px"


    plot.text(x='x', y='y', text='labels', level='glyph',
                      text_baseline='middle', source=source)#, render_mode='canvas', text_alpha=1)


    script, div = components(plot)
    return script, div




@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/search")
def search():
    return render_template("index.html")



def get_ids_from_youtube(search_topic, max_num_videos=50):
    
    url = f'http://www.youtube.com/results?search_query={search_topic}'
    

    chrome_options = Options()
    chrome_options.add_argument('--headless')
    

    chrome_options.add_argument("window-size=1920,1080")
    driver = webdriver.Chrome(options=chrome_options)

    
    
    driver.implicitly_wait(10)
    
    driver.get(url)
    
    
    
    
    links = []

    while len(links) < max_num_videos:
        
        # find element with the video info
        video_elements = driver.find_elements_by_id(id_='video-title')
    
        # TODO: make this more efficient so we don't
        # re-populate the list of videos
        links = []
        for element in video_elements:
            # extract the 'href' tha contains the videoID
            # and the title of the video
            link = element.get_attribute('href')
            title = element.get_attribute('title')
            if link == None:
                continue
            videoId = link.split('v=')[1]
            links.append( (videoId, title) )
            

        # simulate scrolling until we get the 
        # number of videos that we want = max_num_videos
        time.sleep(2)
        driver.execute_script("arguments[0].scrollIntoView();", video_elements[-1])
        time.sleep(2)
    
    
    driver.close()
    
    return links



if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    app.run(host="127.0.0.1", port=5000, debug=True)#, dev_tools_hot_reload=False)
