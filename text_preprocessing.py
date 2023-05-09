#Data analysis libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')

#Data Preprocessing and Feature Engineering
from textblob import TextBlob
import re
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Model Selection and Validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score


#Import data
input_csv = pd.read_csv('output.csv', encoding='ISO-8859-1')

#Parse csv's to only extract column called 'message'
messages = input_csv['MsgBody']

#Text pre-processing functions
def text_processing(message):
    
    #Generating the list of words in the message (hastags and other punctuations removed) and convert to lowercase
    def form_sentence(message):
        message = message.lower() #Make messages lowercase 
        message_blob = TextBlob(message.lower()) #Convert to 'textblob' which provides a simple API for NLP tasks
        return ' '.join(message_blob.words)
    new_message = form_sentence(message)
    
    #Removing stopwords and words with unusual symbols
    def no_user_alpha(message):
        message_list = [item for item in message.split()] 
        clean_words = [word for word in message_list if re.match(r'[^\W\d]*$', word)] #remove punctuation and strange characters
        clean_sentence = ' '.join(clean_words) 
        clean_mess = [stopword for stopword in clean_sentence.split() if stopword not in stopwords.words('english')] #remove stopwords
        return clean_mess
    no_punc_message = no_user_alpha(new_message)
    
    #Normalizing the words in messages 
    def normalization(message_list):
        lem = WordNetLemmatizer()
        normalized_message = []
        for word in message_list:
            normalized_text = lem.lemmatize(word,'v') #lemmatize words
            normalized_message.append(normalized_text)
        return normalized_message
    
    
    return normalization(no_punc_message)

#Print to console and write to file
f = open('processed_text.csv','w', encoding='utf8')
for message in messages: 
	message = text_processing(message)
	for term in message:
		f.write(term+" ")
	f.write("\n")

	
