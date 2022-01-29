
import re # regular expressions for parsing
import pandas as pd # pandas for dataframes
import numpy as np # numpy for math
import nltk #natural language toolkit tokenization and stemming 
import matplotlib.pyplot as plt    # matplotlib for plotting
from nltk.corpus import stopwords # stopwords
from nltk.stem import SnowballStemmer # stemming


print(" All necessary Imports completed ....... \n Now loading data ....\n ")

df_data = pd.read_csv('./data/train.csv') #  # read the train data in to dataframe 
df_test= pd.read_csv('./data/test.csv') # # read the test data in to dataframe 
# df_data = pd.read_csv('./test/dirty5.csv') # read the data in to dataframe 
# df_test= pd.read_csv('./test/test_validation_subset5.csv') # load data

print('First five Rows of Train data',df_data.head()) # print the first 5 rows of the Tranning data
print('\n Data loaded successfully .... \n \n Now Optimizing  data ....\n ')

print('Memory usage by Data: ',df_data.memory_usage(deep=True),'\n \n Preety high Right ??   Dont worry we will Reduce it....\n') 

# convert "category " column to Categorial data to save memory space as they are repeated and in fixed no.
df_data['category'] = df_data['category'].astype('category')

# convert "id " and "category_num" column to int32 to save memory space as they are repeadted and in fixed no.
df_data[['id','category_num']] = df_data[['id','category_num']].astype('int32')

print('\n Lets see Memory usage again: \n \n ',df_data.memory_usage(deep=True),' \n  \n See!! Told you we gonna reduce it \n')


print('\n \n Now we will remove the stop words and apply Snowball stemming to the  the data \n \n')

snowball = SnowballStemmer(language='english') # create a snowball stemmer object
STOPWORDS = set(stopwords.words('english')) # load the stopwords
url_pattern = re.compile(r'https?://\S+|www\.\S+') # regular expression for url

# Function to clean text 
def clean_text(text):

    text = url_pattern.sub(r'',text.replace('\n',' ')) # remove URLs and replace newline by space
    text = re.sub(r'\w*\d\w*', '', text) # remove words containing numbers
    text = re.sub('[^a-z\s]', ' ', text.lower()) #replace all non-alphabetical characters with space and lower case
    text = [word for word in text.split() if word not in STOPWORDS] #remove stopwords
    text = [snowball.stem(i) for i in text] # stemming
    text = ' '.join(text ) # join the list of words 
    text = re.sub(r'\b\w{1,2}\b', '', text) #remove words with length less than 2

    return text

print('\n \n Cleaning started for train.csv . Hold on................  it will take time 10-15 minutes \n \n')
df_data['abstract'] = df_data['abstract'].apply(clean_text)

print('\n \n Cleaning completed for train.csv now saving it as train_clean.csv in data folder \n \n')
df_data.to_csv('./data/train_clean.csv', index=False)
# df_data.to_csv('./test/train_clean.csv', index=False)
print('Lets verify if that function works : \n,',df_data.head(),'\n \n All looks good \n \n ') 

print('\n \n Now cleaning test.csv  and saving it as test_clean.csv \n \n Hold on ............ it will take2-3 minutes \n \n')
df_test['abstract'] = df_test['abstract'].apply(clean_text)
df_test.to_csv('./data/test_clean.csv', index=False)
# df_test.to_csv('./test/test_clean.csv', index=False)
print('Lets verify if that function works : \n,',df_test.head(),'\n \n All looks good \n \n ') 
