from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
import pandas as pd
import os

# Set up the paths to your data
data_dir = r"C:/Users/aadis/Desktop/starbucks"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(r"C:/Users/aadis/Desktop/starbucks/processed_text.csv")

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=123).reset_index(drop=True)

# Split the data into train and test sets
train_data = df.iloc[:int(len(df)*0.8)]
test_data = df.iloc[int(len(df)*0.8):]

# Create the train and test directories if they do not exist
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Save the train and test sets to separate text files in their respective directories
with open(os.path.join(train_dir, "train.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(train_data["MsgBody"].tolist()))
with open(os.path.join(test_dir, "test.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(test_data["MsgBody"].tolist()))

# Read the text file and create a DataFrame with a single "DATA_COLUMN"
train_df = pd.read_csv(r'C:/Users/aadis/Desktop/starbucks/train/train.txt', header=None, names=['DATA_COLUMN'])

# Print the first few rows of the DataFrame
print(train_df.head())


# Read the CSV file and create a DataFrame with a single "DATA_COLUMN"
test_df = pd.read_csv(r'C:/Users/aadis/Desktop/starbucks/test/test.txt', header=None, names=['DATA_COLUMN'])

# Print the first few rows of the DataFrame
print(test_df.head())


InputExample(guid=None,
             text_a = "Hello, world",
             text_b = None,
             label = 1)

def convert_data_to_examples(train_df, test_df, DATA_COLUMN, LABEL_COLUMN): 
    train_InputExamples = train_df.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = None), axis = 1)

    validation_InputExamples = test_df.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = None), axis = 1)

    return train_InputExamples, validation_InputExamples

train_InputExamples, validation_InputExamples = convert_data_to_examples(train_df, 
                                                                           test_df, 
                                                                           'DATA_COLUMN', 
                                                                           'LABEL_COLUMN')
def convert_data_to_examples(data_df, DATA_COLUMN):
    return data_df.apply(lambda x: InputExample(guid=None,
                                                text_a=x[DATA_COLUMN],
                                                text_b=None,
                                                label=None), axis=1)


def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, # pads to the right by default # CHECK THIS for pad_to_max_length
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=None
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )

from transformers import DistilBertTokenizerFast


import numpy as np
# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Define a function to convert data to examples that can be used to train the model
def convert_data_to_examples(data_df, DATA_COLUMN):
    
    # Define a generator function that yields each example in the data
    def gen():
        for _, row in data_df.iterrows():
            # Use the tokenizer to encode the input text
            inputs = tokenizer.encode_plus(
                row[DATA_COLUMN],
                add_special_tokens=True,
                max_length=512,  # Limit the maximum length of the input
                pad_to_max_length=True,  # Pad shorter inputs up to the maximum length
                return_attention_mask=True,  # Return an attention mask to tell the model which tokens to pay attention to
                return_token_type_ids=True,  # Return token type IDs to tell the model which tokens belong to which sequence
                truncation=True  # Truncate longer inputs to the maximum length
            )
            # Convert the inputs to NumPy arrays
            input_ids = np.array(inputs['input_ids'], dtype=np.int32)
            attention_mask = np.array(inputs['attention_mask'], dtype=np.int32)
            token_type_ids = np.array(inputs['token_type_ids'], dtype=np.int32)
            # Yield the input and a default label of 0
            yield ({'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}, 0)

    return gen


# Define the training data as a TensorFlow dataset
train_data = tf.data.Dataset.from_generator(
    # Use the generator function to create examples from the training data
    convert_data_to_examples(train_df, 'DATA_COLUMN'), 
    # Define the output types of the examples
    output_types=({
        'input_ids': tf.int32,
        'attention_mask': tf.int32,
        'token_type_ids': tf.int32
    }, tf.int32)
)
# Shuffle the training data and batch it into sets of 32 examples
train_data = train_data.shuffle(100).batch(32).repeat(2)

# Define the validation data as a TensorFlow dataset
validation_data = tf.data.Dataset.from_generator(
    # Use the generator function to create examples from the validation data
    convert_data_to_examples(test_df, 'DATA_COLUMN'), 
    # Define the output types of the examples
    output_types=({
        'input_ids': tf.int32,
        'attention_mask': tf.int32,
        'token_type_ids': tf.int32
    }, tf.int32)
)
# Batch the validation data into sets of 32 examples
validation_data = validation_data.batch(32)



# Define a function to preprocess the data and cast it to float32
def preprocess_data(x, y):
    return (
        {
            'input_ids': tf.cast(x['input_ids'], tf.float32), 
            'attention_mask': tf.cast(x['attention_mask'], tf.float32), 
            'token_type_ids': tf.cast(x['token_type_ids'], tf.float32)
        },
        y
    )

# Preprocess the training data and cast it to float32
train_data = train_data.map(preprocess_data)
# Preprocess the validation data and cast it to float32
validation_data = validation_data.map(preprocess_data)

# Define the model architecture as a TensorFlow sequential model
model = tf.keras.Sequential([
    # Add layers to your model
])

# Compile the model with an Adam optimizer, sparse categorical crossentropy loss function, and accuracy metric
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
)

# Train the model on the training data for 2 epochs, using the validation data for validation
model.fit(train_data, epochs=2, validation_data=validation_data)

from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

# Load the pre-trained model and tokenizer
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Load the csv file with the Reddit data
df = pd.read_csv('processed_text.csv')

# Define a function to preprocess the text
import re
import nltk
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Convert tokens to lowercase
    tokens = [token.lower() for token in tokens]
    # Join tokens back into text
    text = ' '.join(tokens)
    return text

# Define a function to make predictions on a given input sentence
def predict_sentiment(sentence):
    # Preprocess the input sentence
    sentence = preprocess_text(sentence)
    # Filter the dataframe to only include comments containing the input sentence
    filtered_df = df[df['MsgBody'].str.contains(sentence)]
    # Encode the filtered comments using the tokenizer
    tf_batch = tokenizer(list(filtered_df['MsgBody']), max_length=128, padding=True, truncation=True, return_tensors='tf')
    # Make predictions on the filtered comments
    tf_outputs = model(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
    # Calculate the percentage of positive and negative comments
    num_comments = len(tf_predictions)
    num_positive = tf.reduce_sum(tf.argmax(tf_predictions, axis=1)).numpy()
    num_negative = len(tf_predictions) - num_positive
    pct_positive = num_positive / len(tf_predictions) * 100
    pct_negative = num_negative / len(tf_predictions) * 100
    # Print the percentage of positive and negative comments
    print(f"Number of comments containing the input sentence: {num_comments}")
    print(f"Percentage of positive comments: {pct_positive:.2f}%")
    print(f"Percentage of negative comments: {pct_negative:.2f}%")
    

# Take input of the name of the drink
drink_name = input("Enter the name of the drink: ")

# Call the predict_sentiment function with the input drink name
predict_sentiment(drink_name)
    
    



