import pickle
import preprocessor as p
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from TextCleaning import TextCleaning


# MODEL_PATH = "../model/omicron-sentiment-analysis-indo.h5"
# TOKENIZER_PATH = "../model/tokenizer_without_stopwords.pkl"
# MAX_LENGTH = 35


class ModelPipeline:

    def __init__(self, model_path, tokenizer_path, max_length):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length

    def custom_f1(self, y_true, y_pred):
        def recall_m(y_true, y_pred):
            TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

            recall = TP / (Positives + K.epsilon())
            return recall

        def precision_m(y_true, y_pred):
            TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

            precision = TP / (Pred_Positives + K.epsilon())
            return precision

        precision, recall = precision_m(
            y_true, y_pred), recall_m(y_true, y_pred)

        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def load_resources(self):
        self.model = load_model(self.model_path, custom_objects={
                                'custom_f1': self.custom_f1}, compile=False)
        self.tokenizer = pickle.load(open(self.tokenizer_path, "rb"))
        self.max_length = self.max_length

    def preprocessing(self):
        tc = TextCleaning()
        self.text = p.clean(self.text)
        self.text = tc.all_preprocessing(self.text)
        return self

    def feature_extraction(self):
        self.text = self.tokenizer.texts_to_sequences([self.text])
        self.text = pad_sequences(
            self.text, maxlen=self.max_length, padding='post', truncating='post')
        return self

    def threshold_predict(self, proba):
        if proba > 0.7:
            return 1  # positive
        elif proba >= 0.5:
            return 2  # neutral
        else:
            return 0  # negative

    # Entrypoint 1
    def predict_proba(self, text):
        self.text = text
        self.load_resources()
        self.preprocessing()
        self.feature_extraction()
        return self.model.predict(self.text)  # keras predict

    def predict(self, text_proba):
        return self.threshold_predict(self.predict_proba(text_proba))
