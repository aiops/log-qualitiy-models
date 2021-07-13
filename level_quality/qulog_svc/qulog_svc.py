import spacy
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import f1_score

LABEL_MAPPER = {"info": 0, "error":1}
INVERSE_LABEL_MAPPER = {0:"info", 1:"error"}


class QuLog:
    def __init__(self):
        self.name = "QuLog"
        self.log_level = make_pipeline(StandardScaler(), SVC(gamma="auto", kernel="rbf"))
        self.label_mapper = LABEL_MAPPER
        self.inverse_label_mapper = INVERSE_LABEL_MAPPER
        self.nlp = spacy.load('en_core_web_trf')  # alternative use simpler model like en_web_sm (lower on POS by 0.07)
        # self._train_log_level()
        # self._train_lingustic_quality()

    def train_log_level(self, load_train, load_train_labels):
        load_train_, load_train_labels = self.log2vec_log_level(load_train, load_train_labels, no_labels=True)
        # print(load_train_, load_train_labels)
        self.log_level.fit(load_train_, load_train_labels)

    def predict_log_level(self, load_test):
        load_test = self.log2vec_log_level(load_test)
        return self.log_level.predict(load_test)

    def log2vec_log_level(self, log_messages, labels=None, no_labels=None):
        tokenized = []
        for i in range(0, len(log_messages)):
            # print("{}/{}".format(i, len(log_messages)), "-r")
            tokenized.append(self.nlp(log_messages[i])._.trf_data.tensors[-1].ravel())
        if no_labels == None:
            return tokenized
        else:
            labels_tokenized = [self.label_mapper[label] for label in labels]
            return tokenized, labels_tokenized

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
        self.lingustic_quality.fit(load_train, load_train_labels)

    def predict_lingustic_quality(self, load_test):
        load_test = self.pos2vec_lingustic_quality(load_test)
        return self.lingustic_quality.predict(load_test), self.lingustic_quality.predict_proba(load_test)

    def _train_log_level(self):
        data = pd.read_csv("./data/training_data_log_level_pred.csv")
        features, target = data.values[:, 0], data.values[:, 1]
        self.train_log_level(features, target)

    def _train_lingustic_quality(self):
        data = pd.read_csv("./data/training_data_linguistic_quality.csv")
        features, target = data.values[:, 0], data.values[:, 1]
        self.train_lingustic_quality(features, target)




qulog = QuLog()


data = pd.read_csv("./data/training_data_log_level_pred.csv")
f1_score_val_lp = []
f1_score_val_lq = []

# data = data.iloc[:100]

# for x in range(10):
#     print("-------" * 10)
#     print("Evaluating phase {}/10".format(x+1))
#     train_indecies = np.random.randint(0, data.shape[0], size=int(0.7*data.shape[0]))
#     test_indecies = list(set(np.arange(data.shape[0])).difference(set(train_indecies)))
#     features, target = data.values[train_indecies, 0], data.values[train_indecies, 1]
#
#     print("-------" * 10)
#
#     qulog.train_log_level(features, target)
#     predictions = qulog.predict_log_level(data.values[test_indecies, 0])
#
#     f1_score_val_lp.append(f1_score(predictions, [LABEL_MAPPER[x] for x in data.values[test_indecies, 1]]))
#     print("The F1 score is {}".format(f1_score_val_lp[x]))
#     print("-------" * 10)


data = pd.read_csv("./data/training_data_linguistic_quality.csv")

print("-------"*10)
print("-------"*10)
print("-------"*10)
print("LOG QUALITY")
print("-------"*10)
print("-------"*10)



for x in range(1):
    print("Evaluating phase {}/10".format(x+1))
    train_indecies = np.random.randint(0, data.shape[0], size=int(0.7 * data.shape[0]))
    test_indecies = list(set(np.arange(data.shape[0])).difference(set(train_indecies)))
    features, target = data.values[train_indecies, 0], data.values[train_indecies, 1]

    qulog.train_lingustic_quality(features, target.astype("int32"))
    predictions, scores = qulog.predict_lingustic_quality(data.values[test_indecies, 0])
    f1_score_val_lq.append(f1_score(predictions, data.values[test_indecies, 1].astype("int32")))
    print("The F1 score is {}".format(f1_score_val_lq[x]))
    print(scores)
