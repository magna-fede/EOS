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

###OH: What are these?
### could probably be organised as dictionary or list (df_list['1'], df_list['2'])
df_1List = []
df_2List = []

# get results for group 1 (all participants in one json file)
with open('/imaging/hauk/users/fm02/EOS_data/CLOZEresults/Group1/jatos_results_20210826185507.txt') as f:
    for jsonObj in f:
        df_1 = pd.read_json(jsonObj)
        df_1List.append(df_1)

# ... and group 2
# df = pd.read_json('U:/CLOZEresults/Group1/jatos_results_20210812170851.txt')
with open('/imaging/hauk/users/fm02/EOS_data/CLOZEresults/Group2/jatos_results_20210826140139.txt') as f:
    for jsonObj in f:
        df_2 = pd.read_json(jsonObj)
        df_2List.append(df_2)


# Create one big list containing all participants.
# Each pariticpant is a pd_DataFrame containing all the events in the experiment
# (after the instruction alternating rows about cloze and likert(plausibility) trials)
# 

dfList = [*df_1List, *df_2List]

# exclude participants whose responses were not appropriate based on first analysis
###OH: define those participants as list at top of script
dfList.pop(27)
dfList.pop(40)


# initialise some useful lists
###OH: briefly describe what those are
clozeList = []
plausiList = []
bothList = []
predList = []
wordspredicted = []

# save all the possible word IDs getting them from first and last participant - this is because there are different word IDs for words in group 1 and 2
###OH: There is a lot going on in this line; can this be broken down?
all_resp = pd.Series([[] for _ in range(len([*dfList[0]['ID'].dropna(),*dfList[-1]['ID'].dropna()]))],
                     index=[*dfList[0]['ID'].dropna(),*dfList[-1]['ID'].dropna()])

# drop attention checks ('ID' = 999)
all_resp = all_resp.drop(999)

# same for plausibility
all_plausib = pd.Series([[] for _ in range(len([*dfList[0]['ID'].dropna(),*dfList[-1]['ID'].dropna()]))],
                     index=[*dfList[0]['ID'].dropna(),*dfList[-1]['ID'].dropna()])
all_plausib = all_plausib.drop(999)

# loop over each participant
for df in dfList:
###OH: "cloze" changes types several times, I'm assuming that's intended
    # get cloze trials
    cloze = df[df['trial_type']=='cloze']
    # keep relevant columns
    cloze = cloze[['ID','target','response']]
    clozeList.append(cloze)
    
    # save participant responses
    for index, row in cloze.iterrows():
        if row['ID']!=999: #ignore the attention checks - all participants responded well enough (already checked)
            all_resp[row['ID']].append(row['response'])
    
    # get survey-likert (aka plausibility) trials
    ###OH: "plausibility" changes type several times
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
    for i in range(len(cloze)):
        # this includes ONLY the first word in the response:
            # accept words starting in capital
            # accepts words followed by '.' or ','
        if (cloze['target'].iloc[i]==re.split(('\.| |,'),cloze['response'].iloc[i][0])[0].lower()):
            wordpredicted.append(cloze[['ID','target']].iloc[i])
            predicted+=1
        # accept misspelled word - counting as misspelled words with
        # levenshtein distance <=2 from the target word
        # (some words such as 'aisle' were frequently misspelled)
        else:
            if levenshtein_distance(cloze['target'].iloc[i],
                     re.split(('\.| |,'),cloze['response'].iloc[i][0])[0].lower())<=2:
                wordpredicted.append(cloze[['ID','target']].iloc[i])
                predicted+=1
    # save number of words predicted (to check each participants' performance)
    predList.append(predicted)
    # and which words were predicted
    wordspredicted.append(wordpredicted)

# create a dataframe where to store cloze probabilities and responses
    
get_cloze = pd.concat([df_1[['ID','target']].dropna(),df_2[['ID','target']].dropna()])
b = np.zeros(shape=(len(get_cloze),1))
get_cloze['cloze'] = b
get_cloze = get_cloze[get_cloze['ID']!=999]

for sub in wordspredicted:
    for word in sub:
        if word['ID']!=999:
            get_cloze['cloze'].loc[get_cloze['ID']==word['ID']] +=1

get_cloze['cloze'] = get_cloze['cloze'].astype(int)

get_cloze = get_cloze.merge(all_resp.rename('all_resps'),
                            left_on='ID',right_on=all_resp.index)

# check how many non-empty responses are present for each target word
get_cloze_denominator = get_cloze['all_resps'].apply(length_nonempty) 
get_cloze['cloze'] = get_cloze['cloze'] / get_cloze_denominator

# add 1 to each value because start counting from 0, but the likert that was displayed started from 1 (to 7)
###OH: all_plausib = all_plausib ?
all_plausib = all_plausib = pd.Series([np.array(x)+1 for x in all_plausib],index=all_plausib.index)
all_plausib = all_plausib.apply(np.mean) # return the mean for each target word

get_cloze = get_cloze.merge(all_plausib.rename('plausibility'),
                            left_on='ID',right_on=all_resp.index)

plt.subplots()
sns.histplot(data=get_cloze,x='cloze',bins=5);

plt.subplots()
sns.histplot(data=get_cloze,x='plausibility',bins=[1,2,3,4,5,6,7]);

# check participants based on number of words predicted
minimum_predicted = np.mean(np.array(predList))-2*(np.std(np.array(predList)))
to_exclude = [idx for idx, x in enumerate(predList) if x<minimum_predicted]

to_exclude, minimum_predicted, np.mean(np.array(predList))


###OH: Could the following be a separate script?

# We might want to exclude participants who predicted less than 2 SD than average, in terms of how many words they predicted.
# Indeed, participant 27 responses were often inappropriate when checked manually (e.g., complaining about the task, rather than performing the task), and they predicted only 24 words (considering that 44.7 answers is 2 SD away from the mean). Participant 40 predicted 37 words, but very often left the response box empty (i.e., no response given - I guess to speed up the experiment).

# ## Now let's look how cloze fits with the previous words characteristics
stimuli = pd.read_excel('C:/Users/User/OwnCloud/Sentences/stimuliWITHsimAoA.xlsx', engine='openpyxl',
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
                                 'BLP_rt',
                                 'BLP_accuracy', 
                                 'similarity', 
                                 'AoA', 
                                 'Predictability'])

stimuli = stimuli.merge(get_cloze[['cloze','plausibility']],
                        left_on='Word',right_on=get_cloze['target'])

FREQ = pd.read_csv("C:/Users/User/OwnCloud/Sentences/SUBTLEX-UK.csv",
                   header = 0, sep=';', decimal = '.',
                   usecols = ["Word", "FreqCount", "LogFreq(Zipf)"])
DomPoS = pd.read_csv("C:/Users/User/OwnCloud/Sentences/SUBTLEX-UK.csv",
                     header = 0, sep=';', decimal = '.',
                     usecols = ["Word","DomPoS"])

Position = pd.Series(index=range(len(stimuli)))

for b in range(len(stimuli)):
    Position[b] = re.split(r'[;,\s]\s*',stimuli.Sentence[b]).index(stimuli.Word[b])+1

stimuli['Position'] = Position    
    
sentnoquote = stimuli[['Word', 'Sentence','Predictability']]
    
documents = []
    
###OH: you use range(len(sentnoquote)) again below
### it might make sense to define idx_noquote = range(len(sentnoquote)) here
for sen in range(len(sentnoquote)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(sentnoquote.Sentence[sen]))
    
    # # remove all single characters
    # document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # # Remove single characters from the start
    # document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Remove single s
    document = re.sub(r'\s+[s]\s+', ' ', document) 
    
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    # document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)
    
tokens = pd.Series(index=range(len(documents)), name='Tokens')
  
for i in range(len(documents)):
    
    tokens[i] = nltk.word_tokenize(documents[i])

sentnoquote = pd.concat([sentnoquote, tokens], axis=1)
  
# list of all the preceding words

precs = pd.Series(index=range(len(sentnoquote)))

for i in range(len(sentnoquote)):
    precs[i] = sentnoquote.Tokens[i][(sentnoquote.Tokens[i].index(sentnoquote.Word[i]))-1]

sentnoquote['Preceding'] = precs
sentnoquote = pd.merge(sentnoquote, FREQ, how='inner', left_on=['Preceding'], right_on=['Word'])
sentnoquote = sentnoquote.drop(columns = 'Word_y')
sentnoquote = sentnoquote.rename(columns={"Word_x" : "Word"})

DomPoS = DomPoS.rename(columns={"Word" : "Preceding"})
sentnoquote = pd.merge(sentnoquote, DomPoS, how='inner', on=['Preceding'])

LENprec = pd.Series(index=range(len(sentnoquote)))
for i in range(len(sentnoquote)):
    LENprec[i] = len(sentnoquote.Preceding[i])
sentnoquote['LENprec'] = LENprec
sentnoquote = sentnoquote.rename(columns={"FreqCount":"PRECEDING_Frequency",
                                           "LogFreq(Zipf)": "PRECEDING_LogFreq(Zipf)"})


###OH: could the following be a separate script?
# calculate and print correlation matrix among variables

corrStimuli = pd.merge(stimuli, sentnoquote[['PRECEDING_Frequency',
                                             "PRECEDING_LogFreq(Zipf)",
                                             "LENprec"]],
                       left_on=stimuli.Word, right_on=sentnoquote.Word)

corrStimuli = corrStimuli[['ConcM', 'LEN', 'UN2_F', 'UN3_F', 'Orth', 'OLD20',
                       'FreqCount', 'LogFreq(Zipf)', 'V_MeanSum',
                       'A_MeanSum', 'mink3_SM', 'Position',
                       'BLP_rt', 'BLP_accuracy', 'similarity', 'AoA', 
                       'PRECEDING_Frequency', "PRECEDING_LogFreq(Zipf)", "LENprec",
                        'Predictability','cloze','plausibility']]

corrmat = corrStimuli.corr()

# corrmat.to_csv(path+'CorrelationMatrix.csv')

corrmatlabels = ["Concreteness", 
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
                 "Similarity",
                 "AgeOfAcquisition",
                 "PRECEDING_Frequency",
                 "PRECEDING_LogFreq(Zipf)",
                 "PRECEDING_#letters",
                 "Predictability",
                 "Cloze Probability",
                 "Plausibility"
                 ]    

# in heatmap, indicate exact number in correlations >0.3
labelscorrmat = corrmat.applymap(lambda v: str(round(v,2)) if (1 > abs(v) > 0.3) else '')
plt.subplots(1, 1, figsize = (15, 10))   
sns.heatmap(corrmat, cmap="Spectral_r", annot=labelscorrmat, 
            annot_kws={'fontsize':9}, fmt='', vmin=-1, vmax=1,
            xticklabels=corrmatlabels,
            yticklabels=corrmatlabels)
plt.title("Correlations matrix - all sentences")

plt.show()


# Predictability (which was the original coding 0 for predictable and 1 for unpredictable sentences) and cloze Probability correlation is -.69

# ## Let's calculate the similarity between the words given as responses by participants and the actual respose presented. 
# 

# This can be seen as the "semantic predictability" but we'll call it "cloze semantic probability".
# If participants responses are more similar to the actual word that will be presented (eg, they may provide a synonym of the word or, sometimes, they might have provided an adjective before a known) than this can be captured in the similarity analysis. As some participants provided more than one words, we will include all the words provided.
# 

import gensim.downloader as api
from gensim.parsing import preprocessing
from nltk.corpus import stopwords
from spellchecker import SpellChecker

wv = api.load('word2vec-google-news-300')

spell = SpellChecker()

df = pd.read_csv('C:/Users/User/OwnCloud/ClozeTask_sharedfolder/UK2US.csv',sep=';')
UK2US_dict = pd.Series(df.US.values,index=df.UK).to_dict()
US2UK_dict = pd.Series(df.UK.values,index=df.US).to_dict()

def replace_all(text):
    for i,w in enumerate(text):
        if w in UK2US_dict:
            text[i] = UK2US_dict[w]
    return text

def clean_resp(resps):
    # import stopwords from nltk
    stop_words = set(stopwords.words('english')) 
    all_trials = []
    for i,trial in enumerate(resps):
        print(i)
        new_trial = []
        
        for s in trial:           
            s = str(s[0])
            s = s.lower()
            s = preprocessing.strip_punctuation(s)
            s = preprocessing.strip_numeric(s)
            # using stop words from nltk because too many in gensim stop_word, including target words
            s = nltk.word_tokenize(s)
            s = replace_all(s)
            s = [word for word in s if not word in stop_words]
            misspelled = spell.unknown(s)
            # as similarity work only with words of known spelling, try to correct all words
            for i,w in enumerate(s):
                if w in misspelled:
                    s[i] = spell.correction(w)
                if not(wv.has_index_for(s[i])): # drop it if you could not correct it - or word2vec similarity will give an error
                        s[i] = ''
            if s!=['']:
                new_trial.append(s)
    
        all_trials.append(new_trial)   
        
    return all_trials

responses = get_cloze['all_resps']
targets = replace_all(get_cloze['target'])

tok = clean_resp(responses)

word_similarities = pd.DataFrame(index=np.arange(400),columns=['Word','Sim'])
word_similarities['Word']=targets

for i,t in enumerate(tok):
    similarities = []
    for list_words in t:
        if list_words:
        	# append word with max similarity in among all the responses which are present also on wv vocabulary
            similarities.append(max([wv.similarity(w,targets[i])
                                     for id_w,w in enumerate(list_words)
                                     if wv.has_index_for(w)]))
    
    word_similarities['Sim'].loc[word_similarities['Word']==targets[i]] = np.mean(similarities)


plt.subplots()
sns.histplot(data=word_similarities,x='Sim',bins=10);

for target in targets:
    if target in US2UK_dict:
        word_similarities['Word'][word_similarities['Word']==target] = US2UK_dict[target]
# get back to UK spelling


corrStimuli2 = pd.merge(stimuli,word_similarities, on='Word' )

corrStimuli2 = pd.merge(corrStimuli2,sentnoquote[['PRECEDING_Frequency',
                                                  "PRECEDING_LogFreq(Zipf)",
                                                  "LENprec"]],
                        left_on=stimuli.Word, right_on=sentnoquote.Word)

# corrStimuli2.to_excel('C:/Users/User/OwnCloud/Sentences/stimuli_all_onewordsemsim.xlsx',index=False)

# select only numerical  variables to plot them (e.g., no sentence or word identity)
corrStimuli2 = corrStimuli2[['ConcM', 'LEN', 'UN2_F', 'UN3_F', 'Orth', 'OLD20',
                       'FreqCount', 'LogFreq(Zipf)', 'V_MeanSum',
                       'A_MeanSum', 'mink3_SM', 'Position',
                       'BLP_rt', 'BLP_accuracy', 'similarity',  
                       'PRECEDING_Frequency', "PRECEDING_LogFreq(Zipf)", "LENprec",
                        'Predictability','cloze','plausibility','Sim']]
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

