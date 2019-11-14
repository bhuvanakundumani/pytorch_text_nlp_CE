Dataset can be downloaded from - 
https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews. 
https://drive.google.com/file/d/1iCxHFV6Y383pIFt6on8a0ygRRhO0oipH/view?usp=sharing
Original dataset can be downloaded from - http://ai.stanford.edu/~amaas/data/sentiment/


* data_exp.py - For identifying the max length of the sentence.
* data_prep.py - Loads the dataset and splits into train and test set.
    Modify the frac value in `df = df.sample(frac=0.1, random_state=42)` (frac=0.1 loads 10% of the dataset.)
* preprocess.py - Tokenize and apply GloVe embedding on the dataset.
    Modify the values min_freq = int:2 ( to choose the words that are present more than twice in the dataset. You can also choose the vocab size using MAX_VOCAB_SIZE. Modify the code `text.build_vocab(dataset, data_test,min_freq=min_freq)` or `text.build_vocab(dataset, data_test,max_size=MAX_VOCAB_SIZE )`)
* predict.py - For inference 
* plotloss.py - has functions to plot loss vs epochs
* data/ - Has the raw data and processed data
* model_files/ - model files.
* models/ - has LSTM.py, LSTM_Attn.py
* main.py - model training and evaluation.

