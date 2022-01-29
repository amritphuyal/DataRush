import pandas as pd # pandas for dataframes
import numpy as np # numpy for math
from sklearn.model_selection import train_test_split # train_test_split
import matplotlib.pyplot as plt   # matplotlib for plotting
from keras.preprocessing.text import Tokenizer # Tokenizer
from keras.preprocessing.sequence import pad_sequences # pad_sequences
from keras.callbacks import EarlyStopping # EarlyStopping
from keras.models import Sequential # Sequential 
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D # Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import load_model # load_model

print(" \nImporting Cleaned Training data(train_clean.csv) from data folder\n ")

# Reading the training and testing dataset
df_data = pd.read_csv('./data/train_clean.csv')
# df_data= pd.read_csv('./test/clean5.csv')
#convert abstract column to str
df_data['abstract'] = df_data['abstract'].astype(str)
print('First five Rows of Cleaned Train data\n \n ',df_data.head()) 

## Saving memory
df_data['category'] = df_data['category'].astype('category')
df_data[['id','category_num']] = df_data[['id','category_num']].astype('int32')

# # The maximum number of words to be used for training. (most frequent)
MAX_NB_WORDS =5000
# Max number of words in each abstract
MAX_SEQUENCE_LENGTH = 200

tokenizer = Tokenizer(num_words=MAX_NB_WORDS,oov_token = '<UNK>')
tokenizer.fit_on_texts(df_data.abstract) # fit the tokenizer on the abstracts
word_index = tokenizer.word_index
word_counts= tokenizer.word_counts
print('\n Found %s unique tokens.' % len(word_index))


############### counts words in each abstract 
counts=df_data['abstract'].apply(lambda x: len(str(x).split(' ')))
df_train_clean_counts_word= df_data.copy()
df_train_clean_counts_word['counts_word']=counts

#save to csv
df_train_clean_counts_word.to_csv('train_clean_counts_word.csv',index=False)
print('Analyze train_clean_counts_word.csv in root to determine suitable value for MAX_SEQUENCE_LENGTH \n \n ')
######################


############### Word index and Word counts 
#convert dictionary to dataframe
df_word_index=pd.DataFrame.from_dict(word_index, orient='index')
df_word_index.reset_index(inplace=True)
df_word_index.columns=['word','index_number']
#save to csv
df_word_index.to_csv('word_index.csv',index=False)

print('Analyze word_index.csv in root folder to view index number assigned to each word \n \n ')

#convert wordcount ordered dictionary to dataframe
df_word_counts=pd.DataFrame.from_dict(word_counts, orient='index')
df_word_counts.reset_index(inplace=True)
df_word_counts.columns=['word','counts']
#save to csv
df_word_counts.to_csv('word_counts.csv',index=False)

print('Analyze word_counts.csv in root folder to view the number of times a word is repeated and determine the dictionary length MAX_NB_WORDS \n \n ')
#################################

print('\n Tokenizing abstract\n')
X = tokenizer.texts_to_sequences(df_data.abstract)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = pd.get_dummies(df_data['category_num']).values

print('Spliting data into training and test set in ration 9:1\n')
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

###########   MODEL  #############
EMBEDDING_DIM = 200
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1],trainable=True))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(EMBEDDING_DIM, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(Y[1]), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
epochs = 50
batch_size= 512
 ##############################################

 # train the model  on the whole training set with early stopping 
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]) 


# save trained model to disk

print('\n Saving Trained Model to disk\n')

model.save('./Saved_model/model.h5')

print('\n Testing the model on test (split) data\n')
accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show();

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();
