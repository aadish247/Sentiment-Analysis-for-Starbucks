import pandas as pd
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

#Load the pre-trained model and tokenizer
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

#Load the CSV file
csv_file_path = r'C:\Users\aadis\Desktop\starbucks\processed_text.csv'
data = pd.read_csv(csv_file_path)

#Define a list of popular Starbucks drinks
drinks_list = ['coffee', 'latte', 'mocha', 'cappuccino', 'frappuccino', 'espresso', 'macchiato', 'americano', 'tea', 'chai', 'hot chocolate', 'iced coffee', 'iced tea', 'vanilla latte', 'caramel macchiato', 'pumpkin spice latte', 'white chocolate mocha', 'green tea frappuccino', 'matcha latte', 'chocolate chip frappuccino']

#Create a dictionary to store the results
results_dict = {}

#Loop through each drink in the list
for drink_name in drinks_list:
    
    # Filter the comments for the given drink
    drink_comments = data[data['MsgBody'].str.contains(drink_name, case=False)]

    # Check the number of comments for the given drink
    num_comments = len(drink_comments)
    print("Number of comments for {}: {}".format(drink_name, num_comments))

    # Encode the comments using the tokenizer
    tf_batch = tokenizer(list(drink_comments['MsgBody']), max_length=128, padding=True, truncation=True, return_tensors='tf')

    # Make predictions on the comments
    tf_outputs = model(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)

    # Calculate the percentage of positive and negative comments
    predicted_labels = tf.argmax(tf_predictions, axis=1)
    num_positive = np.count_nonzero(predicted_labels)
    num_negative = len(tf_predictions) - num_positive
    percent_positive = round((num_positive / len(tf_predictions)) * 100, 2)
    percent_negative = round((num_negative / len(tf_predictions)) * 100, 2)

    # Store the results in the dictionary
    results_dict[drink_name] = {'num_comments': num_comments, 'percent_positive': percent_positive, 'percent_negative': percent_negative}


#Convert the dictionary to a pandas DataFrame
results_df = pd.DataFrame.from_dict(results_dict, orient='index')

#Display the results
print(results_df)
