import os
import spacy
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import *


#################################################
##### Preprocessing
#################################################


# another method is required if other nlp models are used
def get_trf_embeddings(trf_docs):
    embeddings = []
    for doc in trf_docs:
        embeddings.append(doc._.trf_data.tensors[-1].ravel())
    return embeddings

def get_small_web_embeddings(vec):
    embeddings = []
    for doc in vec:
        embeddings.append(doc.vector)
    return embeddings

def preprocessing(features, targets, nlp, label2id, buffer=True):
    # Buffering this to prevent recalculation
    token_embeddings_file = "./.small_sm_token_embeddings.pkl"
    if buffer and os.path.isfile(token_embeddings_file):
        print("Loading token embeddings from existing buffer")
        token_embeddings = load_pickle(token_embeddings_file)
    else:
        trf_docs = get_docs(nlp, features)
        token_embeddings = get_small_web_embeddings(trf_docs)
        if buffer:
            store_pickle(token_embeddings, token_embeddings_file)
    token_embeddings = np.asarray(token_embeddings)


    word_class_embeddings_file = "./.small_sm_word_class_embeddings.pkl"
    if buffer and os.path.isfile(word_class_embeddings_file):
        print("Loading token embeddings from existing buffer")
        word_class_embeddings = load_pickle(word_class_embeddings_file)
    else:
        word_classses = [" ".join([t.pos_ for t in doc]) for doc in trf_docs]
        word_classs_trf_docs = get_docs(nlp, word_classses)
        word_class_embeddings = get_small_web_embeddings(word_classs_trf_docs)
        if buffer:
            store_pickle(word_class_embeddings, word_class_embeddings_file)
    word_class_embeddings = np.asarray(word_class_embeddings)


    target_ids_file = "./.target_ids.pkl"
    if buffer and os.path.isfile(target_ids_file):
        print("Loading token embeddings from existing buffer")
        target_ids = load_pickle(target_ids_file)
    else:
        # print(targets)
        # target_ids = [label2id[t] for t in targets]
        target_ids = targets
        if buffer:
            store_pickle(target_ids, target_ids_file)
    target_ids = np.asarray(target_ids)
    return token_embeddings, word_class_embeddings, target_ids


# Adjust if needed
data = pd.read_csv("./training_data_linguistic_quality.csv")
features, targets = data.values[:, 0], data.values[:, 1]
targets = targets.astype("int32")

# This should not be changes. Later, copy the model to the respective directory 
# if you want to publish it. Check the readme first.
model_file_path = "./qulog_svc.joblib"

label2id = {"good": 0, "bad":1}
id2label = {0:"good", 1:"bad"}

model = make_pipeline(StandardScaler(), SVC(gamma="auto", kernel="rbf", verbose=True))
nlp = spacy.load('en_core_web_sm')

token_embeddings, word_class_embeddings, target_ids = preprocessing(features, targets, nlp, label2id)

model = fit_model(model, token_embeddings, target_ids)
store_joblib(model, model_file_path)
