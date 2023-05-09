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

# Define the list of coffee ingredients
coffee_ingredients = ['coffee', 'espresso', 'milk', 'sugar', 'cream', 'cinnamon', 'chocolate', 'caramel',
                      'hazelnut','vanilla', 'whipped cream', 'cocoa powder', 'nutmeg', 'ginger', 'cardamom',
                      'mint', 'almond','coconut', 'maple syrup', 'honey', 'chai', 'matcha', 'lavender', 'rose',
                      'salt', 'pepper','nutella', 'butterscotch', 'toffee', 'white chocolate', 'dark chocolate', 
                      'Irish cream', 'amaretto','kahlua', 'brandy', 'rum', 'whiskey', 'vodka', 'baileys', 
                      'coconut milk', 'soy milk', 'almond milk','condensed milk', 'evaporated milk', 
                      'half and half', 'oat milk', 'coconut oil', 'butter', 'olive oil']


# Create a dictionary to store the number of positive comments for each ingredient
positive_counts = {}

# Iterate through the ingredients and filter the comments for each ingredient
for ingredient in coffee_ingredients:
    # Filter the comments for the given ingredient
    ingredient_comments = data[data['MsgBody'].str.contains(ingredient, case=False)]
    # Check if there are any comments for the given ingredient
    if not ingredient_comments.empty:
        # Check the number of comments for the given ingredient
        num_comments = len(ingredient_comments)
        print("Number of comments for {}: {}".format(ingredient, num_comments))
        # Encode the comments using the tokenizer
        tf_batch = tokenizer(list(ingredient_comments['MsgBody']), max_length=128, padding=True, truncation=True, return_tensors='tf')
        # Make predictions on the comments
        tf_outputs = model(tf_batch)
        tf_predictions = tf.nn.softmax(tf_outputs.logits, axis=-1)
        # Calculate the percentage of positive comments
        num_positive = tf.reduce_sum(tf.cast(tf.argmax(tf_predictions, axis=1), tf.float32))

        percent_positive = round((num_positive / len(tf_predictions)) * 100, 2).numpy()

        # Store the number of positive comments in the dictionary
        positive_counts[ingredient] = num_positive
        # Display the percentage of positive comments
        print("Percentage of positive comments for {}: {}%".format(ingredient, percent_positive))

# Sort the ingredients based on their popularity
if positive_counts:
    sorted_ingredients = sorted(positive_counts.items(), key=lambda x: x[1], reverse=True)
    print("Coffee ingredients sorted by popularity:")
    for ingredient in sorted_ingredients:
        print(ingredient[0], ":", ingredient[1])
else:
    print("No comments found for any of the coffee ingredients.")
