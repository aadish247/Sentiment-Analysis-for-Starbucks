import pandas as pd
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# Load the pre-trained model and tokenizer
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Load the CSV file
csv_file_path = r'C:\Users\aadis\Desktop\starbucks\processed_text.csv'
data = pd.read_csv(csv_file_path)

# Ask for the name of the drink
drink_name = input("Enter the name of the drink: ")

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
num_positive = tf.math.count_nonzero(tf.argmax(tf_predictions, axis=1))
num_negative = len(tf_predictions) - num_positive
percent_positive = num_positive / len(tf_predictions) * 100
percent_negative = 100 - percent_positive

# Display the percentage of positive and negative comments
print("Percentage of positive comments for {}: {:.2f}%".format(drink_name, percent_positive))
print("Percentage of negative comments for {}: {:.2f}%".format(drink_name, percent_negative))
