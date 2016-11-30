from flask import Flask, request, render_template
from operator import itemgetter
from sklearn.externals import joblib
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import SGDClassifier
from base64 import b64encode
from urllib import quote
from random import choice
# from pandas import DataFrame
# from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import gensim
import numpy as np
import pickle
import os


plt.style.use('ggplot')

CLINTON = 'Hillary Clinton'
TRUMP = 'Donald Trump'
SANDERS = 'Bernie Sanders'
JOHNSON = 'Gary Johnson'
STEIN = 'Jill Stein'

EXAMPLES = [CLINTON, TRUMP, SANDERS, JOHNSON, STEIN]

app = Flask(__name__)


@app.route('/')
def main_form():
    return render_template("home.html")


@app.route('/submit_text', methods=['POST'])
def submit_textarea():
    text = request.form["text"]
    print(text.encode("utf-8"))
    submit = request.form
    labelled_tweets = pickle.load(app.open_resource("labelled_tweets.p"))
    if "TextNB" in submit:
        # data = DataFrame(labelled_tweets, columns=['tweet', 'class'])
        # data = data.reindex(numpy.random.permutation(data.index))
        # pipeline = Pipeline([
        #     ('tfidf_vectorizer', TfidfVectorizer(min_df=2, max_df=0.9, stop_words='english')),
        #     ('classifier', MultinomialNB(alpha=.01))
        # ])
        # pipeline.fit(data['tweet'].values, data['class'].values)
        # joblib.dump(pipeline, '/Users/Noah/Desktop/MultinomialNB_Election.pkl', compress=True)
        pipeline = joblib.load("/var/www/intro-to-media/intro-to-media/MultinomialNB_Election.pkl")
        result_probabilities = pipeline.predict_proba([text])
        results_dict = {text: {CLINTON: result[3] * 100,
                               TRUMP: result[1] * 100,
                               SANDERS: result[0] * 100,
                               JOHNSON: result[2] * 100,
                               STEIN: result[4] * 100} for result in result_probabilities}
        pie = bake_pie(results_dict)
        final_results = results_dict[text]
        winner = max(final_results, key=final_results.get)
        winner_score = final_results[winner]
        del final_results[winner]
        text = "\"" + text + "\""
        result = ["is %.2f%s like %s." % (winner_score, "%", winner)]
        for candidate, score in sorted(final_results.items(), key=itemgetter(1), reverse=True):
            result.append(candidate + ": %.2f%s\n" % (score, "%"))
        result.append(pie)
        final_list = [text] + result
        final_list.append(topic_model(labelled_tweets, text, winner))
        return render_template("submit.html", items=final_list)
    elif "TextSGD" in submit:
        # data = DataFrame(labelled_tweets, columns=['tweet', 'class'])
        # data = data.reindex(numpy.random.permutation(data.index))
        # pipeline = Pipeline([
        #     ('tfidf_vectorizer', TfidfVectorizer(min_df=2, max_df=0.9, stop_words='english')),
        #     ('classifier', SGDClassifier(loss='modified_huber', alpha=.0001, n_iter=200, penalty="elasticnet"))
        # ])
        # pipeline.fit(data['tweet'].values, data['class'].values)
        # joblib.dump(pipeline, '/Users/Noah/Desktop/SGDClassifier_modified_huber_elasticnet_Election.pkl', compress=True)
        pipeline = joblib.load("/var/www/intro-to-media/intro-to-media/SGDClassifier_modified_huber_elasticnet_Election.pkl")
        result_probabilities = pipeline.predict_proba([text])
        results_dict = {text: {CLINTON: result[3] * 100,
                               TRUMP: result[1] * 100,
                               SANDERS: result[0] * 100,
                               JOHNSON: result[2] * 100,
                               STEIN: result[4] * 100} for result in result_probabilities}
        pie = bake_pie(results_dict)
        final_results = results_dict[text]
        winner = max(final_results, key=final_results.get)
        winner_score = final_results[winner]
        del final_results[winner]
        text = "\"" + text + "\""
        result = ["is %.2f%s like %s." % (winner_score, "%", winner)]
        for candidate, score in sorted(final_results.items(), key=itemgetter(1), reverse=True):
            result.append(candidate + ": %.2f%s\n" % (score, "%"))
        result.append(pie)
        final_list = [text] + result
        final_list.append(topic_model(labelled_tweets, text, winner))
        return render_template("submit.html", items=final_list)
    return render_template("try_again.html")


@app.route('/submit_selection', methods=['POST'])
def submit_dropdown():
    text = request.form["choice"]
    labelled_tweets = pickle.load(app.open_resource("labelled_tweets.p"))
    if text == "Surprise me!":
        text = choice(labelled_tweets)[0]
    print(text.encode("utf-8"))
    submit = request.form
    if "SelectNB" in submit:
        pipeline = joblib.load("/var/www/intro-to-media/intro-to-media/MultinomialNB_Election.pkl")
        result_probabilities = pipeline.predict_proba([text])
        results_dict = {text: {CLINTON: result[3] * 100,
                               TRUMP: result[1] * 100,
                               SANDERS: result[0] * 100,
                               JOHNSON: result[2] * 100,
                               STEIN: result[4] * 100} for result in result_probabilities}
        pie = bake_pie(results_dict)
        final_results = results_dict[text]
        winner = max(final_results, key=final_results.get)
        winner_score = final_results[winner]
        del final_results[winner]
        text = "\"" + text + "\""
        result = ["is %.2f%s like %s." % (winner_score, "%", winner)]
        for candidate, score in sorted(final_results.items(), key=itemgetter(1), reverse=True):
            result.append(candidate + ": %.2f%s\n" % (score, "%"))
        result.append(pie)
        final_list = [text] + result
        final_list.append(topic_model(labelled_tweets, text, winner))
        return render_template("submit.html", items=final_list)
    elif "SelectSGD" in submit:
        pipeline = joblib.load("/var/www/intro-to-media/intro-to-media/SGDClassifier_modified_huber_elasticnet_Election.pkl")
        result_probabilities = pipeline.predict_proba([text])
        results_dict = {text: {CLINTON: result[3] * 100,
                               TRUMP: result[1] * 100,
                               SANDERS: result[0] * 100,
                               JOHNSON: result[2] * 100,
                               STEIN: result[4] * 100} for result in result_probabilities}
        pie = bake_pie(results_dict)
        final_results = results_dict[text]
        winner = max(final_results, key=final_results.get)
        winner_score = final_results[winner]
        del final_results[winner]
        text = "\"" + text + "\""
        result = ["is %.2f%s like %s." % (winner_score, "%", winner)]
        for candidate, score in sorted(final_results.items(), key=itemgetter(1), reverse=True):
            result.append(candidate + ": %.2f%s\n" % (score, "%"))
        result.append(pie)
        final_list = [text] + result
        final_list.append(topic_model(labelled_tweets, text, winner))
        return render_template("submit.html", items=final_list)
    return render_template("try_again.html")


def bake_pie(results_dict):
    text = "\"" + list(results_dict.keys())[0] + "\""
    final_results = results_dict[text[1:-1]]
    sorted_final_results = sorted(final_results.items(), key=itemgetter(1), reverse=True)
    new_results1 = []
    new_results2 = []
    for candidate, score in sorted_final_results:
        if candidate == CLINTON:
            new_results1.append((score, 'darkblue',))
            new_results2.append((CLINTON, 'darkblue',))
        elif candidate == TRUMP:
            new_results1.append((score, 'darkred',))
            new_results2.append((TRUMP, 'darkred',))
        elif candidate == SANDERS:
            new_results1.append((score, 'dodgerblue',))
            new_results2.append((SANDERS, 'dodgerblue',))
        elif candidate == JOHNSON:
            new_results1.append((score, 'gold',))
            new_results2.append((JOHNSON, 'gold',))
        elif candidate == STEIN:
            new_results1.append((score, 'darkgreen',))
            new_results2.append((STEIN, 'darkgreen',))
        else:
            continue
    useful_list = sorted(new_results1, key=lambda x: x[1][0], reverse=True)
    useful_dict = dict([(color, candidate,) for candidate, color in new_results2])
    sizes = [tup[0] for tup in useful_list]
    colors = [tup[1] for tup in useful_list]
    labels = [useful_dict[color] + " (%1.1f%%)" % size for color, size in zip(colors, sizes)]
    max_index, max_value = max(enumerate(sizes), key=itemgetter(1))
    explode = [0, 0, 0, 0, 0]
    explode[max_index] = 0.075
    explode = tuple(explode)
    patches, texts = plt.pie(sizes, explode=explode, colors=colors, shadow=True, startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.autoscale()
    plt.axis('equal')
    plt.savefig('/var/www/intro-to-media/intro-to-media/pie.png')
    plt.clf()
    final_string = quote(b64encode(open("/var/www/intro-to-media/intro-to-media/pie.png", "rb").read()))
    os.remove("/var/www/intro-to-media/intro-to-media/pie.png")
    return final_string


def topic_model(labelled_tweets, text, winner):
    # def print_top_words(models, feature_names_lists, n_top_words=15):
    #     for model, feature_names in zip(models, feature_names_lists):
    #         for topic_idx, topic in enumerate(model.components_):
    #             print("Topic #%d:" % topic_idx)
    #             print(", ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    #         print("\n")

    dataset_clinton = [tup[0] for tup in labelled_tweets if tup[1] == CLINTON]
    dataset_trump = [tup[0] for tup in labelled_tweets if tup[1] == TRUMP]
    dataset_sanders = [tup[0] for tup in labelled_tweets if tup[1] == SANDERS]
    dataset_johnson = [tup[0] for tup in labelled_tweets if tup[1] == JOHNSON]
    dataset_stein = [tup[0] for tup in labelled_tweets if tup[1] == STEIN]

    datasets = [dataset_clinton, dataset_trump, dataset_sanders, dataset_johnson, dataset_stein]

    datasets = [dataset + [text] for dataset in datasets]

    names = [CLINTON, TRUMP, SANDERS, JOHNSON, STEIN]

    doctopics = []
    # ldas = []
    # vocabs = []
    for j, dataset in enumerate(datasets):
        vectorizer = TfidfVectorizer(min_df=2, max_df=0.9, stop_words='english')
        dtm = vectorizer.fit_transform(dataset).toarray()
        # vocab = vectorizer.get_feature_names()
        # vocabs.append(vocab)
        clf = LatentDirichletAllocation(n_topics=5,
                                        max_iter=1,
                                        learning_method='online',
                                        learning_offset=25.0,
                                        random_state=42)
        # ldas.append(clf)
        doctopic = clf.fit_transform(dtm)
        doctopics.append(doctopic)
        # clf.fit(dtm)

    # print_top_words(ldas, vocabs)

    partial_doctopic = doctopics[0][-1]
    partial_doctopic = np.vstack((partial_doctopic, doctopics[1][-1]))
    partial_doctopic = np.vstack((partial_doctopic, doctopics[2][-1]))
    partial_doctopic = np.vstack((partial_doctopic, doctopics[3][-1]))
    final_doctopic = np.vstack((partial_doctopic, doctopics[4][-1]))
    height, length = final_doctopic.shape
    ind = np.arange(height)
    width = 0.5
    plots = []
    height_cumulative = np.zeros(height)
    for k in range(length):
        div = float(k/float(length))
        color = plt.cm.YlOrRd(div)
        if k == 0:
            p = plt.bar(ind, final_doctopic[:, k], width, color=color)
        else:
            p = plt.bar(ind, final_doctopic[:, k], width, bottom=height_cumulative, color=color)
        height_cumulative += final_doctopic[:, k]
        plots.append(p)
    plt.ylim((0, 1))
    plt.ylabel('Topics')
    plt.title('Topics in ' + text[:60])
    plt.xticks(ind + width / 2, [name + "\n(as shown)" if name == winner else name for name in names])
    plt.yticks(np.arange(0, 1, 10))
    topic_labels = ['Topic #{}'.format(k) for k in range(height)]
    plt.legend([p[0] for p in plots], topic_labels)
    plt.savefig("/var/www/intro-to-media/intro-to-media/bars.png")
    plt.clf()
    final_string = quote(b64encode(open("/var/www/intro-to-media/intro-to-media/bars.png", "rb").read()))
    os.remove("/var/www/intro-to-media/intro-to-media/bars.png")
    return final_string


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
