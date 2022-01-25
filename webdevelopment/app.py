from flask import Flask, render_template, request
from ModelPipeline import ModelPipeline
import time

app = Flask(__name__, template_folder="templates")


def process_ml(teks):
    MODEL_PATH = "./model/omicron-sentiment-analysis-indo.h5"
    TOKENIZER_PATH = "./model/tokenizer_without_stopwords.pkl"
    MAX_LENGTH = 35
    mp = ModelPipeline(MODEL_PATH, TOKENIZER_PATH, MAX_LENGTH)
    prediction = mp.predict(teks)
    if prediction == 1:
        return "Positive"
    elif prediction == 2:
        return "Neutral"
    else:
        return "Negative"


@app.route('/', methods=['GET', 'POST'])
def index():
    time_now = time.time()
    answer = "Belum Ada"
    if request.method == 'POST':
        input_teks = request.form['input']
        answer = process_ml(input_teks)
    time_end = time.time() - time_now
    time_end = '{:.3f}'.format(time_end)
    return render_template("index.html", p=answer, t=time_end)


if __name__ == '__main__':
    app.run(debug=True)
