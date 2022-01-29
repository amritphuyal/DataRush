import pandas as pd # pandas for dataframes
import numpy as np # numpy for math
from keras.preprocessing.text import Tokenizer # Tokenizer
from keras.preprocessing.sequence import pad_sequences # pad_sequence
from keras.models import load_model # load_model

print(" \nImporting Cleaned Test data(test_clean.csv) from data folder\n ")

df_test=pd.read_csv('./data/test_clean.csv')
# df_test= pd.read_csv('./test/test_validation_subset5.csv')
#convert abstract column to str
df_test['abstract'] = df_test['abstract'].astype(str)
print('First five Rows of Cleaned test data \n \n ',df_test.head()) 

new_model = load_model('./Saved_model/model.h5')
print('\n Model import successful \n ')

column_names = ['id','category_num']
df_solution = pd.DataFrame(columns = column_names)

## copy 'id' column of test.csv to "Id" column of df_solution
df_solution.id= df_test.id   # or use below method

print("Prediction start .......... ..... .... ... ")

MAX_NB_WORDS =50000
MAX_SEQUENCE_LENGTH = 200
tokenizer = Tokenizer(num_words=MAX_NB_WORDS,oov_token = '<UNK>') 
tokenizer.fit_on_texts(df_test.abstract) # fit the tokenizer on the abstracts
tok = tokenizer.texts_to_sequences(df_test.abstract)
pad = pad_sequences(tok, maxlen=MAX_SEQUENCE_LENGTH)
solution_series=new_model.predict(pad)
print("Prediction Complete")

Solution_np=np.argmax(solution_series,axis=1)
df_solution['category_num'] =Solution_np
df_solution.to_csv('solution.csv',index=False)
print('Top 10 prediction \n ',df_solution.head(20))
