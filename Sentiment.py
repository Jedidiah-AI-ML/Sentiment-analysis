#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk


# In[65]:


df = pd.read_csv('../sentiment_analysis/Reviews.csv')


# In[66]:


df.head()


# In[67]:


# lets reduce our data to 500 Rows to avoid dealing with large dataset
df.shape


# In[68]:


df = df.head(500)
df.shape


# ## Quick EDA (Exploratary Data Aaylysis)

# In[69]:


df_EDA=df['Score'].value_counts().sort_index()
df_EDA


# In[70]:


df_EDA.plot(kind='bar', title='COunnt of reviews by stars',
           figsize=(10,5))
plt.xlabel('Review Stars')
plt.show()


# ## Basic NLTK 

# In[71]:


example = df['Text'][50]
example


# In[72]:


import nltk
nltk.download('punkt_tab')


# In[73]:


tokens = nltk.word_tokenize(example)
tokens[:10]


# In[74]:


nltk.download('averaged_perceptron_tagger_eng')


# In[75]:


# part of speech for each of these tokes (POS)
tagged=nltk.pos_tag(tokens)
tagged[:10]
# there is a page that cotais the differet POS meaning(i.e NN- a singular noun,)


# In[76]:


nltk.download('maxent_ne_chunker_tab')


# In[77]:


nltk.download('words')


# # putting these tagged into entities
# entities= nltk.chunk.ne_chunk(tagged)
# entities.pprint()

# ## now lets begin
# 
# ## Step 1: VADER sentiment scoring
# we will use NLTK's SentimentIntensityAnalyzer to get the neg/neu/pos scores of the text.
# This uses a "bag of words" approach:
# 1. Stop words are removed(And, or etc because they really do not have any emotions t them)
# 2. each word is scored and combined to the total score

# In[78]:


nltk.download('vader_lexicon')


# In[79]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()


# In[80]:


sia.polarity_scores('i am so happy')


# In[81]:


sia.polarity_scores('today is the worst day ever')


# In[82]:


sia.polarity_scores(example)


# In[83]:


# run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid=row['Id']
    res[myid] = sia.polarity_scores(text)


# In[84]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={"index": "Id"})
vaders = vaders.merge(df, how='left')


# In[85]:


# ow we have setimet score ad metadata
vaders.head()


# ## Plot VADER results

# In[86]:


import seaborn as sns


# In[87]:


ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound Score by Amazonn Star Review')
plt.show()


# In[88]:


# TO look at the pos, eg ad neutral scores of each
sns.barplot(data=vaders, x = 'Score', y='pos')


# In[89]:


fig,axs = plt.subplots(1,3, figsize=(15,5))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# As the star review gets higher, the comments tends to be more positive, neutral comments varry while negative comments tends to decrease significantly

# **human language depends on context(if its a sacarsm, the tone i which it is said)** 

# ### Step 3. Roberta Pretrained Model
# 
# * use a model trained of a large corpus of data.
# * Transformer model accounts for the words ut also the context related to other words

# In[90]:


get_ipython().system('pip install transformers')


# In[91]:


get_ipython().system('pip install transformers')


# In[92]:


pip install torch


# In[93]:


pip install tensorflow


# In[94]:


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import torch.nn.functional as F
import numpy as np


# In[95]:


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[96]:


# VADER results o example
print(example)
sia.polarity_scores(example)


# In[97]:


# Run for Roerta Model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
# chage it to umpy so we ca store it locally
scores = output[0][0].detach().numpy()
scores = softmax(scores)
# make a score dictionary where we can store this
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos': scores[2]
}
scores_dict


# In[98]:


# lets run this on the full data
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    # chage it to umpy so we ca store it locally
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    # make a score dictionary where we can store this
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict


# In[99]:


# run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        
        text = row['Text']
        myid=row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"]=value

        roberta_result = polarity_scores_roberta(text)
        # add the 2 dictionaries together
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f"broke for id {myid}")


    


# In[100]:


results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')


# In[101]:


results_df.head()


# ## Compare scores betweeen models

# In[102]:


results_df.columns


# In[103]:


sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()


# ## Review Examples
# 
# * positive 1-star and negative 5-star Reviews
# 
# lets look at some examples where the model scoring and review score differ the most

# In[104]:


results_df.query('Score ==1') \
    .sort_values('roberta_pos',ascending=False)['Text'].values[0]


# In[105]:


results_df.query('Score ==1') \
    .sort_values('vader_pos',ascending=False)['Text'].values[0]


# In[106]:


results_df.query('Score ==5') \
    .sort_values('roberta_neg',ascending=False)['Text'].values[0]


# In[107]:


results_df.query('Score ==5') \
    .sort_values('vader_neg',ascending=False)['Text'].values[0]


# In[ ]:




