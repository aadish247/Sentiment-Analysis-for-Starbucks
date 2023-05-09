import pandas as pd #Provides text processing capabilities
import numpy as np #Provides Python with better math processing capabilities
from sklearn.feature_extraction.text import CountVectorizer

#The next line of code reads your Reddit data into this program's memory
#Place your reddit data into the same directory of this script and change the below filename
reviews_datasets = pd.read_csv('processed_text.csv')

reviews_datasets = reviews_datasets.head(20000) #The 20,000 number listed as a parameter here is a limitor of how many records you want to analyze. Adjust this number according to the size of your dataset and whether you run into memory limitations
reviews_datasets.dropna() #Drops any records that have a missing value

reviews_datasets.head() #Print first 5 rows to console inspect data 

#This specifies which column to extract for text analysis. It is referenced again a few lines from this comment (doc_term_matrix = count_vect...)
reviews_datasets['MsgBody'][10]

count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english') #Hyperparameters; max_df = maximum document frequency; min_df = minimum document frequency, stop words = 'english')
doc_term_matrix = count_vect.fit_transform(reviews_datasets['MsgBody'].values.astype('U')) #Create document-term matrix
doc_term_matrix

from sklearn.decomposition import LatentDirichletAllocation #Import LDA

#n_components is how many topics you want to generate. 
#This is one of the "hyperparameters" for LDA
#Many machine learning models have similar hyperparameters
#You can adjust hyperparameters to tune model performance
LDA = LatentDirichletAllocation(n_components=10, random_state=42) #n_components = number of topics to generate; random_state = a seed to produce reproducible results
#More documentation here: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
LDA.fit(doc_term_matrix)

first_topic = LDA.components_[0]

top_topic_words = first_topic.argsort()[-10:]
       
#Prints out the most "important" words for forming topic distribution     
print("Most \"Important\" words for forming topic distribution")  
for i in top_topic_words:
    print(count_vect.get_feature_names()[i])
    

for i,topic in enumerate(LDA.components_):
    print(f'Top 10 words for coffee topic #{i}:')
    print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')
    
topic_values = LDA.transform(doc_term_matrix)
topic_values.shape