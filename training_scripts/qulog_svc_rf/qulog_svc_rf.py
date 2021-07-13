import os
import pickle
from joblib import dump
import spacy
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score

LABEL2ID = {"info": 0, "error":1}
ID2LABEL = {0:"info", 1:"error"}


class QuLog:
    def __init__(self, label2id=LABEL2ID, id2label=ID2LABEL):
        # Models
        self.model_log_level = make_pipeline(StandardScaler(), SVC(gamma="auto", kernel="rbf", verbose=True))
        self.model_ling_quality = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, min_samples_split=2))

        # Label mapping
        self.label2id = label2id
        self.id2label = id2label

        # NLP model
        self.nlp = spacy.load('en_core_web_trf')

    def fit_log_level(self, features, targets):
        self.model_log_level.fit(features, targets)

    def predict_log_level(self, features):
        return self.model_log_level.predict(features)

    def get_trf_docs(self, log_messages):    
        print("Calculating embeddings...")
        trf_docs = list(self.nlp.pipe(log_messages))
        print("Embedding calculation done.")
        return trf_docs
        
    def labels2ids(self, labels):    
        return [self.label2id[label] for label in labels]

    def get_embeddings(self, trf_docs):
        embeddings = []
        for doc in trf_docs:
            embeddings.append(doc._.trf_data.tensors[-1].ravel())
        return embeddings

    def prepare_linguistic(self, embeddings):
        pos_data = []
        for embedding in embeddings:
            pos_data.append([token.pos_ for token in embedding])
                
            #pos_data.append(self.nlp(" ".join(pos_desc))._.trf_data.tensors[-1].ravel())


    def pos2vec_lingustic_quality(self, log_messages):
        data_pos = []
        for log in log_messages:
            log = self.nlp(log)
            pos_desc = []
            for token in log:
                pos_desc.append(token.pos_)
            data_pos.append(self.nlp(" ".join(pos_desc))._.trf_data.tensors[-1].ravel())
            # print(self.nlp(" ".join(pos_desc))._.trf_data.tensors[-1].ravel().shape)
            #
        return data_pos

    def train_lingustic_quality(self, load_train, load_train_labels):
        load_train = self.pos2vec_lingustic_quality(load_train)
        self.model_ling_quality.fit(load_train, load_train_labels)

    def predict_lingustic_quality(self, load_test):
        load_test = self.pos2vec_lingustic_quality(load_test)
        return self.model_ling_quality.predict(load_test), self.model_ling_quality.predict_proba(load_test)

    def _train_log_level(self):
        data = pd.read_csv("./training_data_log_level_pred.csv")
        features, target = data.values[:, 0], data.values[:, 1]
        self.train_log_level(features, target)

    def _train_lingustic_quality(self):
        data = pd.read_csv("./training_data_linguistic_quality.csv")
        features, target = data.values[:, 0], data.values[:, 1]
        self.train_lingustic_quality(features, target)


def store_pickle(object, file_path):
    with open(file_path, 'wb') as fh:
        pickle.dump(object, fh)

def load_pickle(file_path):
    with open(file_path, 'rb') as fh:
        object = pickle.load(fh)
    return object


qulog = QuLog()

data = pd.read_csv("./training_data_log_level_pred.csv")

#################################################
##### Preprocessing
#################################################

features, target = data.values[:, 0], data.values[:, 1]

token_embeddings_file = "./trf_docs.pkl"
if os.path.isfile(token_embeddings_file):
    token_embeddings = load_pickle(token_embeddings_file)
else:
    trf_docs = qulog.get_trf_docs(features)
    token_embeddings = qulog.get_embeddings(trf_docs)
    store_pickle(token_embeddings, token_embeddings_file)
token_embeddings = np.asarray(token_embeddings)


word_class_embeddings_file = "./word_class_trf_docs.pkl"
if os.path.isfile(word_class_embeddings_file):
    word_class_embeddings = taget_ids = load_pickle(word_class_embeddings_file)
else:
    word_classses = [" ".join([t.pos_ for t in doc]) for doc in trf_docs]
    word_classs_trf_docs = qulog.get_trf_docs(word_classses)
    word_class_embeddings = qulog.get_embeddings(word_classs_trf_docs)
    store_pickle(word_class_embeddings, word_class_embeddings_file)
word_class_embeddings = np.asarray(word_class_embeddings)


target_ids_file = "./target_ids.pkl"
if os.path.isfile(target_ids_file):
    target_ids = load_pickle(target_ids_file)
else:
    target_ids = qulog.labels2ids(target)
    store_pickle(target_ids, target_ids_file)
target_ids = np.asarray(target_ids)


#################################################
##### Log Level - Evaluation
################################################# 

def evaluate(token_embeddings, target_ids):
    f1_scores = []

    sample_size = len(taget_ids)
    for i in range(10):
        print("-------" * 10)
        print("Evaluating phase {} / 10".format(i+1))
        train_indecies = np.random.randint(0, sample_size, size=int(0.7*sample_size))
        test_indecies = list(set(np.arange(sample_size)).difference(set(train_indecies)))
        train_x, train_y = token_embeddings[train_indecies, :], target_ids[train_indecies]
        qulog.fit_log_level(train_x, train_y)

        print("-------" * 10)

        test_x, test_y = token_embeddings[test_indecies], target_ids[test_indecies]
        pred_y = qulog.predict_log_level(test_x)
        f1_scores.append(f1_score(pred_y, test_y))

        print("The F1 score is {}".format(f1_scores[i]))
        print("-------" * 10)

#evaluate(token_embeddings, target_ids)

#################################################
##### Log Level - Train & Store
#################################################

print("-------" * 10)
print("Training log level quality checking model...")

qulog.fit_log_level(token_embeddings, target_ids)

print("Training done.")
print("-------" * 10)

model_file_path = "../../level_quality/qulog_svc.joblib"
print("Storing model as {}...".format(model_file_path))

dump(qulog.model_log_level, model_file_path)

print("Model storing done.")