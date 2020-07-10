#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd 
from collections import Counter
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t')

def preprocess(df):
    df['question1'] = df['question1'].apply(lambda x: x.lower())
    df['question2'] = df['question2'].apply(lambda x: x.lower())
    return df

df_duplicates = df[df['is_duplicate'] == 1]
df_duplicates = preprocess(df_duplicates)


train, test = train_test_split(df_duplicates, train_size=0.9, random_state=1234)    


train.to_csv('data/quora/train.csv')
test.to_csv('data/quora/test.csv')



