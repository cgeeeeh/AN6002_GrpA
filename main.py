
from nltk.stem import PorterStemmer
ps=PorterStemmer()
print(ps.stem('speaking'))


from nltk.stem import WordNetLemmatizer
wnl=WordNetLemmatizer()
print(wnl.lemmatize('speaking','v'))

from nltk.stem import PorterStemmer
ps=PorterStemmer()
print(ps.stem('supporters'))

from nltk.stem import WordNetLemmatizer
wnl=WordNetLemmatizer()
print(wnl.lemmatize('supporters','n'))

##campaigning
ps=PorterStemmer()
print(ps.stem('campaigning'))


wnl=WordNetLemmatizer()
print(wnl.lemmatize('campaigning','v'))


from nltk import word_tokenize
with open("candidate1.txt", "r", encoding ="utf-8") as f:
    words= word_tokenize(f.read())
    different_words=[]
    for word in words:
        if ps.stem(word) != wnl.lemmatize(word):
            different_words.append((ps.stem(word), wnl.lemmatize(word)))
    print(different_words)
    print(len(different_words))


##part-of-speech (POS) tagging example
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
wnl=WordNetLemmatizer()
from nltk import pos_tag
words = ['speaking','supporter','campaigning']
print(pos_tag(words))
print(wnl.lemmatize(words[0],'v'))
print(wnl.lemmatize(words[1],'n'))
print(wnl.lemmatize(words[2],'v'))

##Task 2
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
df=pd.read_csv("news.csv")
analyzer = SentimentIntensityAnalyzer()
print(df.iloc[0,-1])
positive_news = []
negative_news=[]
neutral_news=[]
print(analyzer.polarity_scores(df.iloc[0,-1])['compound'])
for index, row in df.iterrows():
    score =analyzer.polarity_scores(row[-1])['compound']
    if score>= 0.05:
        positive_news.append(row[-1])
    elif score <=-0.05:
        negative_news.append(row[-1])
    else:
        neutral_news.append(row[-1])

print(len(positive_news), len(negative_news), len(neutral_news))
