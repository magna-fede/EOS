# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:16:05 2023

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

path = "C:/Users/fm02/OwnCloud/Sentences/"

os.chdir(path)

conc = pd.read_csv("concreteness.csv", header=0, decimal = ',',
                   usecols = ["Word", "Conc.M"])
FREQ = pd.read_excel(r"C:\Users\fm02\Downloads\SUBTLEX-US.xlsx", header = 0,  engine="openpyxl", \
                   usecols = ["Word", "FREQcount", "Zipf-value"])
data = data[data['Word_Content_Or_Function'] == "Content"]
cols = ['Participant_ID', 'Word_Unique_ID', "Word", "Word_Length","OrthographicMatch", "LSA_Context_Score",
        "Word_Number", "Word_POS", "Word_Content_Or_Function", "IA_FIRST_FIX_PROGRESSIVE",
        "POS_CLAWS", "IA_FIRST_FIXATION_DURATION","IA_FIRST_RUN_DWELL_TIME"]

data = data[cols]
data = data[data["IA_FIRST_FIX_PROGRESSIVE"]==1.0]
data = pd.merge(data, FREQ, how='inner', on=['Word'])
data = pd.merge(data, conc, how='inner', on=['Word'])


[sns.distplot(data['OrthographicMatch'][data['Word_POS']==key], label=key) for key in ['Adjective', 'Adverb', 'Noun', 'Verb']]
plt.legend()
plt.show()

[sns.distplot(data['LSA_Context_Score'][data['Word_POS']==key], label=key) for key in ['Adjective', 'Adverb', 'Noun', 'Verb']]
plt.legend()
plt.show()

[sns.distplot(data['Word_Length'][data['Word_POS']==key], label=key) for key in ['Adjective', 'Adverb', 'Noun', 'Verb']]
plt.legend()
plt.show()

[sns.distplot(data['Zipf-value'][data['Word_POS']==key], label=key) for key in ['Adjective', 'Adverb', 'Noun', 'Verb']]
plt.legend()
plt.show()

uniq = data.drop_duplicates(subset=['Word'], keep='first')
scaler = StandardScaler()
provo_normalised = data.copy()

for col in ['Word_Length', 'OrthographicMatch', 'FREQcount', 'Zipf-value', 'Conc.M', 'LSA_Context_Score']:
    scaler.fit(np.array(uniq[col]).reshape(-1,1))
    provo_normalised[col] = scaler.transform(np.array(provo_normalised[col]).reshape(-1,1))


import gensim.downloader as api
from gensim.parsing import preprocessing
from nltk.corpus import stopwords
from spellchecker import SpellChecker

# load w2v trained on google news
wv = api.load('word2vec-google-news-300')

# intialised spell check
spell = SpellChecker()

# import dataframe with two columns: one has US spelling, the other UK
# this is necessary because word2vec has words in US spelling and not UK
df = pd.read_csv('C:/Users/fm02/OwnCloud/ClozeTask_sharedfolder/UK2US.csv',sep=';')
UK2US_dict = pd.Series(df.US.values,index=df.UK).to_dict()
US2UK_dict = pd.Series(df.UK.values,index=df.US).to_dict()

responses = pd.read_csv(r"C:\Users\fm02\Downloads\Provo_Corpus-Predictability_Norms.csv",
                        encoding='ISO-8859-1')

targets = list()
resp = list()
word_ids = responses['Word_Unique_ID'].unique()

for word_id in word_ids:
    print(word_id)
    a = responses[['Word', 'Response', 'Response_Count']][responses['Word_Unique_ID']==word_id]
    b = a.apply(lambda x: [x['Response']]*x['Response_Count'], axis=1)
    c = [item for sublist in b for item in sublist]
    resp.append(c)
    targets.append(a['Word'].unique()[0])

sims = pd.DataFrame(columns=['Word_Unique_ID', 'Predictabilty'])

for i, word_id in enumerate(word_ids):
    if word_id in provo_normalised['Word_Unique_ID'].values:
        if wv.has_index_for(targets[i]):
            sim = np.array([wv.similarity(targets[i], w) for w in resp[i] if wv.has_index_for(w)]).mean()
            row = pd.DataFrame({'Word_Unique_ID': word_id, 'Predictabilty': sim}, index=[0])
            sims = pd.concat([sims, row], axis=0, ignore_index=True)
        else:
            row = pd.DataFrame({'Word_Unique_ID': word_id, 'Predictabilty': np.nan}, index=[0])
            sims = pd.concat([sims, row], axis=0, ignore_index=True)

sims_NOTnorm = sims.copy()
sims['Predictabilty'] = scaler.fit_transform(np.array(sims['Predictabilty']).reshape(-1, 1))
provo_normalised_pred = provo_normalised.merge(sims, on="Word_Unique_ID")
data = data.merge(sims_NOTnorm, on="Word_Unique_ID")

provo_normalised_pred.to_csv(r'C:\Users\fm02\ownCloud\Manuscripts\EOS\provo_normalised_pred.csv', index=False)

stimuli = pd.read_excel('C:/Users/fm02/OwnCloud/Sentences/stimuli_ALL.xlsx', engine='openpyxl')
stimuli = stimuli[['LogFreq(Zipf)', 'Position', 'LEN', 'Sim', 'cloze', 'ConcM']]


corrmatlabels2 = ["LogFreq(Zipf)",  
                 "PositionSent", 
                 "#letters",                
                 "LSA Context Scores",
                 "Predictability",
                 "Cloze_Probability",
                 "Concreteness", 
                 ] 

corr = provo_normalised_pred[['Zipf-value', 
                              'Word_Number',
                              'Word_Length',
                              'LSA_Context_Score',
                              'Predictabilty',
                              'OrthographicMatch',
                              'Conc.M']].corr()

sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})

sns.set_theme(context="notebook",
              style="white",
              font="sans-serif")

sns.set_style("ticks")

colors = sns.color_palette(["#FFBE0B",
                            #"#FB5607",
                            "#FF006E",
                            "#8338EC",
                            "#3A86FF",
                            "#1D437F"
                            ])
customPalette = sns.set_palette(sns.color_palette(colors))

for col, name, stim_col in zip(['Zipf-value', 
            'Word_Number',
            'Word_Length',
            'LSA_Context_Score',
            'Predictabilty',
            'OrthographicMatch',
            'Conc.M'], ["LogFreq(Zipf)",  
                             "PositionSent", 
                             "#letters",                
                             "LSA Context Scores",
                             "Predictability",
                             "Cloze_Probability",
                             "Concreteness", 
                             ],
            ['LogFreq(Zipf)', 'Position', 'LEN', '', 'Sim', 'cloze', 'ConcM']):
                    
    sns.histplot(data, stat='density', common_norm=False, kde_kws=dict(cut=0),
                 x=col, kde=True, hue='Word_POS', bins=30, palette=customPalette,
                 alpha=0.1, line_kws=dict(linewidth=2))
    if stim_col:
        sns.kdeplot(data=stimuli, linewidth=2, cut=0,
                 x=stim_col, label='EOS', color="#1D437F")
    plt.legend(['Nouns', 'Adjectives', 'Adverbs', 'Verbs', 'EOS'])
    plt.xlabel(name)
    plt.show()
    

labelscorrmat2 = corr.applymap(lambda v: str(round(v,2)) if (1 > abs(v) > 0.2) else '')
plt.subplots(1, 1, figsize = (12, 8))   
sns.heatmap(corr, cmap="Spectral_r", annot=labelscorrmat2, 
            annot_kws={'fontsize':15}, fmt='', vmin=-1, vmax=1,
            xticklabels=corrmatlabels2,
           yticklabels=corrmatlabels2)
plt.title("Correlations matrix - all sentences")

plt.show()

d = data[['Word_Unique_ID', 'Word_Length', 'OrthographicMatch', 'Word_Number', 'Zipf-value', 'Conc.M', 'LSA_Context_Score']]
d = pd.merge(d, sims, on='Word_Unique_ID')
