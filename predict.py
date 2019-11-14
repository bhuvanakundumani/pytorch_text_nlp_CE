import torch
from models.Crossentropy.LSTM import LstmClassifier
import numpy as np 
from nltk import word_tokenize

# load model
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model

def classify(text: str):
    tokens = word_tokenize(text)
    fixed_text = " ".join(tokens)
    tokens = [word2idx.get(token, 0) for token in tokens]
    tokens = torch.tensor(tokens).expand(1,-1)
    #seq_len = torch.tensor([len(tokens)])
    label = classifier(tokens)
    label = torch.max(label,1)[1]
    print(label)
    return label

def label_conv(label):
    print("within funct", label)
    return idx2label.get(label, 'Invalid label')

word2idx = torch.load("model_files/word2index.pkl")
idx2label = np.load("model_files/label2idx.npy", allow_pickle=True).item()

classifier = LstmClassifier(1,2,100,25518,100, False, pretrained_embed=False)
classifier = load_model(classifier,"model_files/model.pkl")

## Sample predictions
label1 = classify("This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues")
print("Sentiment is", label_conv(int(label1)))

label2 = classify("what a horrible movie that was")
print("Sentiment is", label_conv(int(label2)))

