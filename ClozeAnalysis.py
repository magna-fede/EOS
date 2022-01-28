#!/usr/bin/env python
# coding: utf-8

import nltk
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.metrics.distance import edit_distance as levenshtein_distance

# we will use this function to calculate the number of non-empty responses for each target word
def length_nonempty(df_list):
    length = len(df_list)
    empties = df_list.count([""])
    return length-empties



# exclude participants whose responses were not appropriate based on first analysis
exclude = [27, 40]

# save temporarily participants data in two different lists, corresponding to the two groups
# faster to incorprorate them in one big list later
df_1List = []
df_2List = []

# get results for group 1 (all participants in one json file)
# with open('C:/Users/User/OwnCloud/ClozeTask_sharedfolder/CLOZEresults/Group1/jatos_results_20210826185507.txt') as f:
with open('//cbsu/data/Imaging/hauk/users/fm02/EOS_data/CLOZEresults/Group1/jatos_results_20210826185507.txt') as f:
    for jsonObj in f:
        df_1 = pd.read_json(jsonObj)
        df_1List.append(df_1)

# ... and group 2
#with open('C:/Users/User/OwnCloud/ClozeTask_sharedfolder/CLOZEresults/Group2/jatos_results_20210826140139.txt') as f:
with open('//cbsu/data/Imaging/hauk/users/fm02/EOS_data/CLOZEresults/Group2/jatos_results_20210826140139.txt') as f:
    for jsonObj in f:
        df_2 = pd.read_json(jsonObj)
        df_2List.append(df_2)


# Create one big list containing all participants.
# Each pariticpant is a pd.DataFrame containing all the events in the experiment
# (after the instruction alternating rows about cloze and likert(plausibility) trials)

dfList = [*df_1List, *df_2List]


for ex in exclude:
	dfList.pop(ex)

# initialise some useful lists

# in this list we will save each participant responses to the cloze task
clozeList = []
# in this plausibility ratings
plausiList = []
# here we save the number of words predicted fro each participant 
# (to check performance)
predList = []
# here we save each word correctly predicted by the participants
wordspredicted = []

# save all the possible word IDs getting them from first and last participant - this is because there are different word IDs for words in group 1 and 2

# this creates a Series, with index=IDs, and each value as an empty list
# thi empty list will be populated with all the possible responses later
all_resp = pd.Series([[] for _ in range(len([*dfList[0]['ID'].dropna(),
                                             *dfList[-1]['ID'].dropna()]))],
                     index=[*dfList[0]['ID'].dropna().astype(int),
                            *dfList[-1]['ID'].dropna().astype(int)])

# drop attention checks ('ID' = 999)
all_resp = all_resp.drop(999)

# same for plausibility
all_plausib = pd.Series([[] for _ in range(len([*dfList[0]['ID'].dropna(),
                                                *dfList[-1]['ID'].dropna()]))],
                     index=[*dfList[0]['ID'].dropna().astype(int),
                            *dfList[-1]['ID'].dropna().astype(int)])
all_plausib = all_plausib.drop(999)

# loop over each participant
for df in dfList:

    # get cloze trials
    cloze = df[df['trial_type']=='cloze']
    # keep relevant columns
    cloze = cloze[['ID','target','response']]
    # save cloze trials in the appropriate list, which will be used to calculate
    # each word's cloze probability
    clozeList.append(cloze)
    
    # save participant responses
    for index, row in cloze.iterrows():
        if row['ID']!=999: #ignore the attention checks - all participants responded well enough (already checked)
            all_resp[row['ID']].append(row['response'])
    
    # get survey-likert (aka plausibility) trials

    plausibility = df[df['trial_type']=='survey-likert']
    plausibility = plausibility[['response']]
    plausibility = plausibility.rename(columns={'response':'plausibility'})
    # assign the same ID (because it's the same order - each plausibility rating follows the order of the same)
    plausibility['ID'] = cloze['ID'].tolist()
    plausiList.append(plausibility)
    
    # save all plausibilities in one place (used to calculate average)
    for index, row in plausibility.iterrows():
        if row['ID']!=999:
            all_plausib[row['ID']].append(row['plausibility'].get('Q0'))
    
    wordpredicted = []           
    predicted = 0  
    
    # now let's count how many words did each participant predict
    # 
    for i in range(len(cloze)):
        # this includes ONLY the first word in the response:
            # accept words starting in capital
            # accepts words followed by '.' or ','
        if (cloze['target'].iloc[i] == 
            re.split(('\.| |,'),cloze['response'].iloc[i][0])[0].lower()):
            
            wordpredicted.append(cloze['ID'].iloc[i].astype(int))
            predicted+=1
        # accept misspelled word - counting as misspelled words with
        # levenshtein distance <=2 from the target word
        # (some words such as 'aisle' were frequently misspelled)
        else:
            if levenshtein_distance(cloze['target'].iloc[i],
                     re.split(('\.| |,'),cloze['response'].iloc[i][0])[0].lower())<=2:
                wordpredicted.append(cloze['ID'].iloc[i].astype(int))
                predicted+=1
    # save number of words predicted (to check each participants' performance)
    predList.append(predicted)
    # and which words were predicted
    wordspredicted.append(wordpredicted)

# create a dataframe where to store cloze probabilities and responses
    
get_cloze = pd.concat([df_1[['ID','target']].dropna(),df_2[['ID','target']].dropna()])
get_cloze['cloze'] = np.zeros(shape=(len(get_cloze),1))
get_cloze['ID'] = get_cloze['ID'].astype(int)
get_cloze = get_cloze.set_index('ID')

for sub in wordspredicted:
    for word in sub:
        if word!=999:
            get_cloze['cloze'].loc[word] +=1

get_cloze['cloze'] = get_cloze['cloze'].astype(int)

get_cloze = get_cloze.merge(all_resp.rename('all_resps'),
                            left_on='ID',right_on=all_resp.index)

# check how many non-empty responses are present for each target word
get_cloze_denominator = get_cloze['all_resps'].apply(length_nonempty) 
get_cloze['cloze'] = get_cloze['cloze'] / get_cloze_denominator

# add 1 to each value because start counting from 0, but the likert that was displayed started from 1 (to 7)

all_plausib = pd.Series([np.array(x)+1 for x in all_plausib],index=all_plausib.index)
all_plausib = all_plausib.apply(np.mean) # return the mean for each target word

get_cloze = get_cloze.merge(all_plausib.rename('plausibility'),
                            left_on='ID',right_on=all_resp.index)

plt.subplots()
sns.histplot(data=get_cloze,x='cloze',bins=5);

plt.subplots()
sns.histplot(data=get_cloze,x='plausibility',bins=[1,2,3,4,5,6,7]);


##############################################################################
##############################################################################
### DIFFERENT PART OF THE SCRIPT, LET's CALUCLATE CLOZE_SEMANTIC_SIMILARITY
##############################################################################
##############################################################################

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

def replace_all(text, Udict):
    """"Replace elements using UK2US or US2UK in a nested structure"""
    text2 = text.copy()
    for i,w in enumerate(text2):
        if w in Udict:
            text2[i] = Udict[w]
    return text2 

def clean_resp(resps):
    """This function checks that responses are ready to be feed to word2vec."""
    # import stopwords from nltk
    stop_words = set(stopwords.words('english')) 
    all_trials = []
    # loop over all trials(i.e., wordIDs)
    for i,trial in enumerate(resps):
        # check at what point are we at
        print(i)
        new_trial = []
        # loop over responses (from different participants)        
        for s in trial:           
            # convert list (each response is contained inside a list) to string
            s = str(s[0])
            s = s.lower()
            s = preprocessing.strip_punctuation(s)
            s = preprocessing.strip_numeric(s)
            # using stop words from nltk because too many in gensim stop_word, including target words
            s = nltk.word_tokenize(s)
            s = replace_all(s, UK2US_dict)
            # (re)create a list of words for each response
            s = [word for word in s if not word in stop_words]
            # find words of unknown spelling
            misspelled = spell.unknown(s)
            # as similarity works only with words of known spelling, try to correct all words
            for i,w in enumerate(s):
                if w in misspelled:
                    # try to correct misspelled words
                    s[i] = spell.correction(w)
                # if a word is not included in w2v dictionary drop it
                # or w2v will throw an error
                if not(wv.has_index_for(s[i])): 
                        s[i] = ''
            if s!=['']:
                new_trial.append(s)
    
        all_trials.append(new_trial)   
        
    return all_trials

responses = get_cloze['all_resps']
targets = replace_all(get_cloze['target'], UK2US_dict)

# prepare responses to be analysed in w2v (consider that w2v is US not UK)
tok = clean_resp(responses)

word_similarities = pd.DataFrame(index=np.arange(400),columns=['Word','Sim'])
word_similarities['Word']=targets

# for each trial (i.e., wordID)
for i,t in enumerate(tok):
    similarities = []
    # for each response in that trial
    for list_words in t:
        # check this is not empty
        if list_words:
            # calculare similarities with target word and append the word that
            # is most similar to the target word.
            similarities.append(max([wv.similarity(w,targets[i])
                                     for w in list_words if wv.has_index_for(w)]))
    # calculate now the average similarity between responses and the target word
    # this is our measure of CLOZE SEMANTIC SIMILARITY
    word_similarities['Sim'].loc[word_similarities['Word']==targets[i]] = np.mean(similarities)

# distribution is less skewed the normal cloze distribution
plt.subplots()
sns.histplot(data=word_similarities,x='Sim',bins=10);

# get back to UK spelling
word_similarities['Word'] = replace_all(word_similarities['Word'], US2UK_dict)

# ## Now let's look how cloze fits with the previous words characteristics
# first, load the 
stimuli = pd.read_excel('C:/Users/fm02/OwnCloud/Sentences/stimuli_ALL.xlsx', engine='openpyxl',
                        usecols=['ID',
                                 'Word',
                                 'Sentence',
                                 'ConcM',
                                 'LEN', 
                                 'UN2_F', 
                                 'UN3_F', 
                                 'Orth', 
                                 'OLD20',
                                 'FreqCount', 
                                 'LogFreq(Zipf)', 
                                 'V_MeanSum',
                                 'A_MeanSum', 
                                 'mink3_SM', 
                                 'Position',
                                 'BLP_rt',
                                 'BLP_accuracy', 
                                 'similarity',
                                 'PRECEDING_Frequency', 
                                 "PRECEDING_LogFreq(Zipf)", 
                                 "LENprec",
                                 'AoA', 
                                 'Predictability'])

stimuli = stimuli.merge(get_cloze[['cloze','plausibility']],
                        left_on='Word',right_on=get_cloze['target'])


corrStimuli2 = pd.merge(stimuli,word_similarities, on='Word' )

# corrStimuli2.to_excel('C:/Users/fm02/OwnCloud/Sentences/stimuli_all_onewordsemsim.xlsx',index=False)

corrStimuli2 = corrStimuli2[['ConcM', 
                             'LEN', 
                             'UN2_F', 
                             'UN3_F', 
                             'Orth', 
                             'OLD20',
                             'FreqCount', 
                             'LogFreq(Zipf)', 
                             'V_MeanSum',
                             'A_MeanSum', 
                             'mink3_SM', 
                             'Position',
                             'BLP_rt', 
                             'BLP_accuracy', 
                             'similarity',  
                             'PRECEDING_Frequency', 
                             "PRECEDING_LogFreq(Zipf)", 
                             "LENprec",
                             'Predictability',
                             'cloze',
                             'plausibility',
                             'Sim']]
corrStimuli2['Sim'] = pd.to_numeric(corrStimuli2['Sim'])
corrmat2 = corrStimuli2.corr()

corrmatlabels2 = ["Concreteness", 
                 "#letters", 
                 "BigramFreq", 
                 "TrigramFreq", 
                 "OrthN",
                 "OLD20",
                 "Frequency", 
                 "LogFreq(Zipf)", 
                 "Valence", 
                 "Arousal", 
                 "SensorimotorStrength", 
                 "PositionSent",
                 "BLP_rt",
                 "BLP_accuracy",
                 "Semantic_Similarity_CONTEXT",
                 "PRECEDING_Frequency",
                 "PRECEDING_LogFreq(Zipf)",
                 "PRECEDING_#letters",
                 "Apriori_Predictability",
                 "Cloze_Probability",
                 "Plausibility",
                 "Semantic_Similarity_CLOZE"
                 ]    

# in heatmap, indicate exact number in correlations >0.3
labelscorrmat2 = corrmat2.applymap(lambda v: str(round(v,2)) if (1 > abs(v) > 0.3) else '')
plt.subplots(1, 1, figsize = (15, 10))   
sns.heatmap(corrmat2, cmap="Spectral_r", annot=labelscorrmat2, 
            annot_kws={'fontsize':9}, fmt='', vmin=-1, vmax=1,
            xticklabels=corrmatlabels2,
            yticklabels=corrmatlabels2)
plt.title("Correlations matrix - all sentences")

plt.show()

