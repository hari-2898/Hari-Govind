
import nltk
from nltk import word_tokenize, pos_tag, bigrams, trigrams, RegexpParser
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist, LidstoneProbDist
from collections import Counter

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

text = "Tomatoes are essential in cooking. They grow well in warm climates."
print("Text:", text)

tokens = word_tokenize(text)
print("\nTokens:", tokens)

words = [w for w in tokens if w.lower() not in stopwords.words('english') and w.isalpha()]
print("\nStop Words Removed:", words)

stemmer = PorterStemmer()
print("\nStemming:", [stemmer.stem(w) for w in words])

tags = pos_tag(tokens)
print("\nPOS Tags:", tags)
print("POS Counts:", Counter(t for w, t in tags))

bi = list(bigrams(tokens))
tri = list(trigrams(tokens))
print("\nBigrams:", bi)
print("\nTrigrams:", tri)

fd = FreqDist(bi)
print("\nBigram Frequency:", fd)

prob = LidstoneProbDist(fd, 0.1, bins=len(fd))
print("\nProbability:", prob.prob(('Tomatoes', 'are')))

grammar = "NP: {<DT>?<JJ>*<NN>|<NNS>}"
tree = RegexpParser(grammar).parse(tags)

print("\nChunking:")
print(tree)

