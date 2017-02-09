"""
have page where text is enterd
create summery
count words
machine learning to create random text
"""
from flask import Flask, render_template, request, redirect, url_for
import nltk
from nltk.corpus import stopwords
import markovify
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


app = Flask(__name__)

def prepare_data(text):
    default_stopwords = set(nltk.corpus.stopwords.words('english'))
    words = nltk.word_tokenize(text)
    # Remove single-character tokens (mostly punctuation)
    words = [word for word in words if len(word) > 1]
    # Remove numbers
    words = [word for word in words if not word.isnumeric()]
    # Lowercase all words (default_stopwords are lowercase too)
    words = [word.lower() for word in words]
    # remove stopwords
    words = [word for word in words if word not in default_stopwords]


    return words

def get_stats(text):
    words = prepare_data(text)
    num_words = len(words)
    num_chars = len(text)
    fdist = nltk.FreqDist(words)
    f_dist = fdist.most_common(10)
    return (num_chars, num_words, f_dist)


def get_summary(text):
    parser = PlaintextParser(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()

    summary = summarizer(parser.document, 3) #Summarize the document with 5 sentences
    paragraph = ""
    for sentance in summary:
        paragraph += sentance._text
        paragraph += " "
    return paragraph

    return summary
def markov_chain(text,n):
    text_model = markovify.Text(text)
    paragraph = ""
    for _ in range(n):
        paragraph += text_model.make_sentence()
        paragraph += " "
    return paragraph


@app.route("/")
def get_text():
    return render_template("get_text.html")

@app.route("/send_text/", methods=['POST'])
def send_text():
    text_inputted=request.form['main-text']
    num_chars, num_words, f_dist = get_stats(text_inputted)
    return render_template("stats.html",
                           chars = num_chars,
                           words = num_words,
                           f_dist = f_dist,
                           markov_chain = markov_chain(text_inputted, 6),
                           summary = get_summary(text_inputted)
                           )
    # return render_template('stats.html', summary = text_inputted)


if __name__ == "__main__":
    app.run(debug=True, )
