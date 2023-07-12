# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 16:36:27 2023

@author: fm02
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


data = pd.read_csv(r"C:\Users\fm02\Downloads\Provo_Corpus-Eyetracking_Data.csv")
data.columns
cols = ['Participant_ID', 'Word_Unique_ID', "Word", "Word_Length","OrthographicMatch",
        "Word_Number", "Word_POS", "Word_Content_Or_Function", "IA_FIRST_FIX_PROGRESSIVE",
        "POS_CLAWS", "IA_FIRST_FIXATION_DURATION","IA_FIRST_RUN_DWELL_TIME"]

data = data[cols]

path = "C:/Users/fm02/OwnCloud/Sentences/"

os.chdir(path)

conc = pd.read_csv("concreteness.csv", header=0, decimal = ',',
                   usecols = ["Word", "Conc.M"])
FREQ = pd.read_excel(r"C:\Users\fm02\Downloads\SUBTLEX-US.xlsx", header = 0,  engine="openpyxl", \
                   usecols = ["Word", "FREQcount", "Zipf-value"])

data['POS_CLAWS'].value_counts()
nouns = ['NN1', 'NN2', 'NN0', 'NP0']

data = data[data['POS_CLAWS'].isin(nouns)]
# equivalent to data = data[data['Word_POS'] == "Noun"]
data['POS_CLAWS'].value_counts()

# NN1    38052
# NN2    14028
# NP0     3360
# NN0      588

data = data.drop(columns=['POS_CLAWS'])

data = pd.merge(data, FREQ, how='inner', on=['Word'])
data = pd.merge(data, conc, how='inner', on=['Word'])

###############################################################################
############################ PLOTS ############################################
###############################################################################


stimuli = pd.read_excel('C:/Users/fm02/OwnCloud/Sentences/stimuli_ALL.xlsx', engine='openpyxl')


stimuli.columns
# Out[57]: 
# Index(['ID', 'Word', 'ConcM', 'V_MeanSum', 'A_MeanSum', 'mink3_SM', 'Sentence',
#        'Predictability', 'BLP_rt', 'BLP_accuracy', 'OLD20', 'LEN', 'Orth',
#        'UN2_F', 'UN3_F', 'FreqCount', 'LogFreq(Zipf)', 'similarity', 'AoA',
#        'cloze', 'plausibility', 'Position', 'Sim', 'PRECEDING_Frequency',
#        'PRECEDING_LogFreq(Zipf)', 'LENprec'],
#       dtype='object')

stimuli = stimuli[['Word', 'ConcM', 'LogFreq(Zipf)', 'LEN', 'cloze', 'Sim' ]]

sns.distplot(stimuli['cloze'], bins=10); sns.distplot(data['OrthographicMatch'], bins=10)
plt.legend(['EOS', 'Provo'])
plt.show()

sns.distplot(stimuli['ConcM'], bins=[1,2,3,4,5]); sns.distplot(data['Conc.M'], bins=[1,2,3,4,5])
plt.legend(['EOS', 'Provo'])
plt.show()

sns.distplot(stimuli['LogFreq(Zipf)'], bins=10); sns.distplot(data['Zipf-value'], bins=10)
plt.legend(['EOS', 'Provo'])
plt.show()

sns.distplot(stimuli['LEN'], hist=False); sns.distplot(data['Word_Length'], hist=False)
plt.legend(['EOS', 'Provo'])
plt.show()

sns.pairplot(data[['Word_Length', 'Zipf-value', 'Conc.M', 'OrthographicMatch']])
plt.title('PROVO')
plt.show()

sns.pairplot(stimuli[['LEN', 'LogFreq(Zipf)', 'ConcM', 'cloze']])
plt.title('EOS')
plt.show()

uniq = data.drop_duplicates(subset=['Word'], keep='first')
scaler = StandardScaler()

provo_normalised = data.copy()

for col in ['Word_Length', 'OrthographicMatch', 'FREQcount', 'Zipf-value', 'Conc.M']:
    scaler.fit(np.array(uniq[col]).reshape(-1,1))
    provo_normalised[col] = scaler.transform(np.array(provo_normalised[col]).reshape(-1,1))

provo_normalised.to_csv(r'C:\Users\fm02\ownCloud\Manuscripts\EOS\provo_normalised.csv', index=False)


##############################################################################
data = pd.read_csv(r"C:\Users\fm02\Downloads\Provo_Corpus-Eyetracking_Data.csv")
data.columns

path = "C:/Users/fm02/OwnCloud/Sentences/"

os.chdir(path)

conc = pd.read_csv("concreteness.csv", header=0, decimal = ',',
                   usecols = ["Word", "Conc.M"])
FREQ = pd.read_excel(r"C:\Users\fm02\Downloads\SUBTLEX-US.xlsx", header = 0,  engine="openpyxl", \
                   usecols = ["Word", "FREQcount", "Zipf-value"])

data['POS_CLAWS'].value_counts()
nouns = ['NN1', 'NN2', 'NN0', 'NP0']

data = data[data['Word_Content_Or_Function'] == "Content"]
# equivalent to data = data[data['Word_POS'] == "Noun"]
data['POS_CLAWS'].value_counts()

# NN1    38052
# NN2    14028
# NP0     3360
# NN0      588

data = data.drop(columns=['POS_CLAWS'])

data = pd.merge(data, FREQ, how='inner', on=['Word'])
data = pd.merge(data, conc, how='inner', on=['Word'])

data = data[data["IA_FIRST_FIX_PROGRESSIVE"]==1.0]

###############################################################################
[sns.distplot(data['Conc.M'][data['Word_POS']==key], label=key) for key in ['Adjective', 'Adverb', 'Noun', 'Verb']]
plt.legend()
plt.show()

[sns.distplot(data['OrthographicMatch'][data['Word_POS']==key], label=key) for key in ['Adjective', 'Adverb', 'Noun', 'Verb']]
plt.legend()
plt.show()

[sns.distplot(data['Word_Length'][data['Word_POS']==key], label=key) for key in ['Adjective', 'Adverb', 'Noun', 'Verb']]
plt.legend()
plt.show()

[sns.distplot(data['Zipf-value'][data['Word_POS']==key], label=key) for key in ['Adjective', 'Adverb', 'Noun', 'Verb']]
plt.legend()
plt.show()

sns.pairplot(stimuli[['LEN', 'LogFreq(Zipf)', 'ConcM', 'cloze']])
plt.title('EOS')
plt.show()

uniq = data.drop_duplicates(subset=['Word'], keep='first')
scaler = StandardScaler()

provo_normalised = data.copy()

for col in ['Word_Length', 'OrthographicMatch', 'FREQcount', 'Zipf-value', 'Conc.M']:
    scaler.fit(np.array(uniq[col]).reshape(-1,1))
    provo_normalised[col] = scaler.transform(np.array(provo_normalised[col]).reshape(-1,1))

provo_normalised.to_csv(r'C:\Users\fm02\ownCloud\Manuscripts\EOS\provo_normalised.csv', index=False)

