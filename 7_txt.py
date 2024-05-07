import pandas as pd
import re
text_data = [
    " This is a sample text with numbers 123 and specia@l characters like !@#$%^&*",
    "Cleaning text is crucial for effective Natural Language Processing",
    "Let's explore how regular expressions can help in text cleaning"
]
text_data
type(text_data)
df = pd.DataFrame({"Text":text_data})
df
df["Text"][0]
df["Cleaned_Text"] = df["Text"].apply(lambda x:re.sub(r"[^A-Za-z0-9\s]", "",x))
df["Cleaned_Text"][0]
df["Cleaned_Text"] = df['Cleaned_Text'].apply(lambda x:x.lower())
df
df["Cleaned_Text"] = df['Cleaned_Text'].apply(lambda x: re.sub(r"\d", "",x))
df
df["Cleaned_Text"] = df["Cleaned_Text"].apply(lambda x: re.sub(r'[^\w\s]',"",x))
df['Text'][0]
df["Cleaned_Text"] = df["Cleaned_Text"].apply(lambda x: re.sub(r'\s+'," ",x).strip())
df
stopword = ["this","is","for","in","with"]
df["Cleaned_Text"] = df["Cleaned_Text"].apply(lambda x: " ".join([word for word in x.split() if word.lower() not in stopword]))
df
df["Text"][0]
df["Cleaned_Text"][0]



## Represent a document in a set of words
"This is a simple example"
[1,1,1,1,1]
import nltk
from sklearn.feature_extraction.text import CountVectorizer
sentences = "This is an example of tutorial. I think this example will be good"
tokens = nltk.word_tokenize(sentences)
tokens
vectorizer = CountVectorizer()
bow_representation = vectorizer.fit_transform([sentences]).toarray()
bow_representation
sent = "I want to visit London as soon as possible because I heard London is a great place"
tokens = nltk.word_tokenize(sent)
tokens
vectorizer = CountVectorizer()
bow = vectorizer.fit_transform([sent]).toarray()
bow


import pandas as pd
## It assigns POS to each word like verb,adj,etc.
import nltk
from nltk.tokenize import word_tokenize
from nltk import  pos_tag
sent = "Natural language processing is amazing!"
words = word_tokenize(sent)
words
pos_tags = pos_tag(words)
pos_tags
data = {"Sentences" : ["NLTK is a powerful tool for NLP.",
        "It provides various functionalities",
        "Turkey is a great country"]}
df = pd.DataFrame(data)
df
data
df["POS_tagging"] = df["Sentences"].apply(lambda x : pos_tag(word_tokenize(x)))
df
df["POS_tagging"][0][0][1]

