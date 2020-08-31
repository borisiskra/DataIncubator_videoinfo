
import glob
import pickle
from collections import Counter

# #### Bert sentence embedding:
from sentence_transformers import SentenceTransformer


import numpy as np
# from numpy import linalg
# from numpy.linalg import norm
# from scipy.spatial.distance import squareform, pdist
from scipy.spatial.distance import sqeuclidean, cosine

# We import sklearn.
# import sklearn
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.manifold import TSNE
# from sklearn.datasets import load_digits
# from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
# from sklearn.metrics.pairwise import pairwise_distances

# from sklearn.manifold.t_sne import (_joint_probabilities,
#                                    _kl_divergence)


# from sklearn.utils.extmath import _ravel
# Random state.
RANDOM_STATE = 42

from scipy.spatial.distance import cdist

# We'll use matplotlib for graphics.
# import matplotlib.pyplot as plt
# import matplotlib.patheffects as PathEffects
# import matplotlib
# get_ipython().run_line_magic('matplotlib', 'inline')

# Porque no bokeh??
# from bokeh.transform import jitter, factor_cmap
# from bokeh.plotting import figure, output_notebook, show, ColumnDataSource, output_file
# from bokeh.models import Legend, Range1d, LabelSet, BoxAnnotation
# from bokeh.io import export_png
# from bokeh.layouts import row, column
# from bokeh.palettes import Category10
# from bokeh.transform import linear_cmap

# import pandas as pd

# We import seaborn to make nice plots.
#import seaborn as sns
#sns.set_style('darkgrid')
#sns.set_palette('muted')
#sns.set_context("notebook", font_scale=1.5,
#                rc={"lines.linewidth": 2.5})



def BERT_embed_topic(topic):

    wiki_fname = f"data/wiki/{topic}_clean.txt"
    full_embedding_fname = f"data/topics/{topic}_full.pickle"
    small_embedding_fname = f"data/topics/{topic}_small.pickle"
    try:
        with open(wiki_fname, 'r') as wiki_file, open(full_embedding_fname, 'wb') as full_embed_file, open(small_embedding_fname, 'wb') as small_embed_file:

            topic_content = wiki_file.readlines()

            embedding_dct = {}
            lst_sentences = []

            for sentence in topic_content:
                sentence = sentence.strip("\n")
                if sentence == '':
                    continue
                print(sentence[:25], '...', sentence[-25:])
                lst_sentences.append(sentence)

            embedding_dct['sentences'] = lst_sentences
            embedding = BERT_embed_sentence(lst_sentences)
            embedding_dct['embedding'] = embedding
            print('Done embeding')
            embedding_dct['centroid'] = centroid(embedding)
            embedding_dct['cosine_radius'] = radius(embedding)
            embedding_dct['cosine_diameter'] = diameter(embedding)
            print('Done cosine distances')
            embedding_dct['euclid_radius'] = radius(embedding, 'euclid')
            embedding_dct['euclid_diameter'] = diameter(embedding, 'euclid')
            embedding_dct['outlier_radius'] = outlier_radius(embedding)

            pickle.dump(embedding_dct, full_embed_file)

            del embedding_dct['sentences']
            del embedding_dct['embedding']

            pickle.dump(embedding_dct, small_embed_file)

    except Exception as err:
        print(f'No wiki file on topic: {topic}')
        print(err)


def BERT_embed_sentence(sentence):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    # print("          BERT")
    embedding = model.encode(sentence)
    # print(len(embedding))
    return embedding


def cosine_dist(vect_1, vect_2):
    return cosine(vect_1.reshape(1, -1), vect_2.reshape(1, -1))


def euclid_dist(vect_1, vect_2):
    # return sqeuclidean(vect_1, vect_2)
    return sqeuclidean(vect_1.reshape(1, -1), vect_2.reshape(1, -1))


def log_cos_dist(vector_1, vector_2):
    return -np.log((2-cosine(vector_1, vector_2))/2)


def diameter(embedding, metric='cosine'):
    max_dist = 0
    farthest_vectors = {}
    for idx_1, vect_1 in enumerate(embedding):
        for idx_2, vect_2 in enumerate(embedding):
            if idx_1 == idx_2:
                continue

            if metric == 'cosine':
                dist = cosine_dist(vect_1, vect_2)
                # print(cos_sim)
            elif metric == 'euclid':
                dist = sqeuclidean(vect_1, vect_2)
            else:
                print('metric not defined')
            if dist > max_dist:
                max_dist = dist
                farthest_vectors[1] = idx_1
                farthest_vectors[2] = idx_2

    return max_dist, farthest_vectors


def centroid(embedding):
    return np.array(embedding).mean(axis=0)


def radius(embedding, metric='cosine', drop_outliers=True):
    r = 0
    center = centroid(embedding)

    for vector in embedding:
        if metric == 'cosine':
            r = max(r, cosine_dist(center, vector))
        elif metric == 'euclid':
            r = max(r, euclid_dist(center, vector))
        else:
            print('metric not defined')
            r = -1
    return r


def outlier_radius(embedding, metric='cosine'):
    radius_lst = []
    center = centroid(embedding)

    for vector in embedding:
        if metric == 'cosine':
            radius_lst.append(cosine_dist(center, vector))
        elif metric == 'euclid':
            radius_lst.append(euclid_dist(center, vector))
        else:
            print('metric not defined')
            r = -1
        r = np.quantile(radius_lst, .5)*4
    return r


def bert_classify_sentences(sentences, test=False):
    # embedding_dct = {}
    # print(type(sentences), len(sentences))
    sentence_embed = BERT_embed_sentence(sentences)
    print(len(sentence_embed), sentence_embed[0].shape)
    # if len(sentence_embed) == 1:
        # test_vector = sentence_embed
    # else:
        # need to change next 3 lines
        # print("Embedding ERROR")
        # print(len(sentence_embed))
        # print(sentences)
        # return
        
    all_embedding = []
    all_topics = []
    for file in glob.glob('./data/topics/*_small.pickle'):
        # print(file)
        topic = file.split('\\')[1].split('_')[0]
        all_topics.append(topic)
        # print(topic)
        # 1/0
        embedding_fname = file
        try:
            with open(embedding_fname, 'rb') as embed_f:
                topic_embedding = pickle.load(embed_f)
        except Exception as err:
            print(err)
            print('No pre embedding found for', topic)
            print(embedding_fname)
            return

        all_embedding.append(topic_embedding['centroid'])
        # print(topic_embedding)
            # embedding_dct[topic]=topic_embedding

    # print(len(all_embedding), all_embedding[0].shape)
    closest_topic = closest(all_embedding, sentence_embed)
    # print('çosine')
    # print(closest(all_embedding, sentence_embed, 'cosine'))
    # print('canberra')
    # print(closest(all_embedding, sentence_embed, 'canberra'))
    # print('log_cos')
    # print(closest(all_embedding, sentence_embed, log_cos_dist))

    # print(closest_topic)
    classification = list(map(lambda i: all_topics[i], closest_topic))

    # print(classification)

    count = Counter(classification)
#    for test_vector in test_embed:
    # print(count)
    # closest_cos=['Unknown',10000,0] # need to change to Infinity
    # closest_euc=['Unknown',11100,0] # need to change to Infinity
    # closest_log=['Unknown',11100,0] # need to change to Infinity

    return count



def closest(topic_embed, sentence_embed, metric='euclidean'):
    dist_matrix = cdist(topic_embed, sentence_embed, metric)
    return np.argmin(dist_matrix, axis=0)
    #return dist_matrix










def bert_test():
    y_true=[]
    y_pred=[]
    all_topics=[]

    embedding_dct = {}
    for topic in TOPICS:
        embedding_fname = "embeddings/{}_bert.pickle".format(topic)
        try:
            with open(embedding_fname, 'rb') as embed_f:
                topic_embedding = pickle.load(embed_f)
        except Exception as err:
            print(err)
            print('No pre embedding found for',topic)
            print(embedding_fname)
            return

        embedding_dct[topic]=topic_embedding
        
    try:
        with open('test/test_data.txt', 'r') as f:
            testing_data_lst = eval(f.read())
            
    except Exception as err:
        print(err)
        print('NO data testing file')
        return

    for test_topic, sentence in testing_data_lst:
        if test_topic == 'Unknown':
            continue
        if test_topic not in all_topics:
            all_topics.append(test_topic)

        result = bert_classify_sentences([sentence], test=True)
        #print(len(result))
        #print(result)
        closest_cos, closest_euc, closest_log = result[0]
        y_true.append(test_topic)
        print("real topic",test_topic)
        print(sentence)
        y_pred.append(closest_cos[0])
        print(closest_cos,closest_euc,closest_log)
        print()
    
    print("y_true")
    print(y_true)
    print("y_pred")
    print(y_pred)
    print("all topics")
    print(all_topics)

    from sklearn.metrics import accuracy_score, f1_score as F1, recall_score
    from sklearn.metrics import precision_recall_fscore_support


    print(precision_recall_fscore_support(y_true, y_pred, average='macro'))

    print(precision_recall_fscore_support(y_true, y_pred, average='micro'))

    print(precision_recall_fscore_support(y_true, y_pred, average='weighted'))
