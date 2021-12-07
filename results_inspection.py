# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:26:35 2021

@author: fm02
"""


import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

base_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/"


with open("U:/AnEyeOnSemantics/30analysis/nrgr_dur_all.P", 'rb') as f:
      ALL_nrgr_ffd = pickle.load(f)
nrgrdur_all = list(ALL_nrgr_ffd.values())

with open("U:/AnEyeOnSemantics/30analysis/wrgr_dur_all.P", 'rb') as f:
      ALL_wrgr_ffd = pickle.load(f)
wrgrdur_all = list(ALL_wrgr_ffd.values())

with open("U:/AnEyeOnSemantics/30analysis/nrgr_ffd_all.P", 'rb') as f:
      ALL_nrgr_ffd = pickle.load(f)
nrgrffd_all = list(ALL_nrgr_ffd.values())
     
with open("U:/AnEyeOnSemantics/30analysis/wrgr_ffd_all.P", 'rb') as f:
      ALL_wrgr_ffd = pickle.load(f)
wrgrffd_all = list(ALL_wrgr_ffd.values())

with open("U:/AnEyeOnSemantics/30analysis/nrgr_gd_all.P", 'rb') as f:
      ALL_nrgr_gd = pickle.load(f)
nrgrgd_all = list(ALL_nrgr_gd.values())

with open("U:/AnEyeOnSemantics/30analysis/wrgr_gd_all.P", 'rb') as f:
      ALL_wrgr_gd = pickle.load(f)
wrgrgd_all = list(ALL_wrgr_gd.values())
     
     
with open("U:/AnEyeOnSemantics/30analysis/norm_nrgr_ffd_all.P", 'rb') as f:
      ALL_norm_nrgr_ffd = pickle.load(f)
norm_nrgrffd_all = list(ALL_norm_nrgr_ffd.values())
     
with open("U:/AnEyeOnSemantics/30analysis/norm_wrgr_ffd_all.P", 'rb') as f:
      ALL_norm_wrgr_ffd = pickle.load(f)
norm_wrgrffd_all = list(ALL_norm_wrgr_ffd.values())

with open("U:/AnEyeOnSemantics/30analysis/norm_nrgr_gd_all.P", 'rb') as f:
      ALL_norm_nrgr_gd = pickle.load(f)
norm_nrgrgd_all = list(ALL_norm_nrgr_gd.values())

with open("U:/AnEyeOnSemantics/30analysis/norm_wrgr_gd_all.P", 'rb') as f:
      ALL_norm_wrgr_gd = pickle.load(f)
norm_wrgrgd_all = list(ALL_norm_wrgr_gd.values())     
     

n_regressions = pd.Series(np.zeros(len(nrgrdur_all),dtype=int))
n_blinks = pd.Series(np.zeros(len(nrgrdur_all),dtype=int))

n_trials_excluded = pd.Series(np.zeros(len(nrgrffd_all)))
n_skipped_fp = pd.Series(np.zeros(len(nrgrffd_all),dtype=int))
percentage_good = pd.Series(np.zeros(len(nrgrffd_all),dtype=int))
avg_ffd = pd.Series(np.zeros(len(nrgrffd_all)))
avg_gd = pd.Series(np.zeros(len(nrgrgd_all)))
n_fixatedfirstpass  = pd.Series(np.zeros(len(nrgrffd_all),dtype=int))

for p,participant in enumerate(nrgrdur_all):
    for trial in participant:
        if type(trial)==str:
            if trial == "Nope - there was regression":
                n_regressions[p] += 1
            elif trial == "There was a blink":
                n_blinks[p] += 1

for p,participant in enumerate(nrgrffd_all):
    n_trials_excluded[p] = len(nrgrdur_all[p]) - len(nrgrffd_all[p])
    n_skipped_fp[p] = len(participant[participant['ms']==0])
    # a trial is good if in ffd, and if ms>0 (means that's included in lmer)
    percentage_good[p] = participant['ms'][participant['ms']>0].count()/400*100
    avg_ffd[p] = np.mean(participant['ms'][participant['ms']!=0])

for p,participant in enumerate(nrgrgd_all):
    avg_gd[p] = np.mean(participant['ms'][participant['ms']!=0])


for i,df, in enumerate(nrgrffd_all):
    nrgrffd_all[i]['Subject'] = [i]*len(nrgrffd_all[i])
nrgrffd_all_long = pd.concat(nrgrffd_all)

ax = sns.boxplot(x="Subject", y="ms", data=nrgrffd_all_long[nrgrffd_all_long['ms']!=0])
plt.title('First Fixation Durations');
plt.show()

for i,df, in enumerate(nrgrgd_all):
    nrgrgd_all[i]['Subject'] = [i]*len(nrgrgd_all[i])
nrgrgd_all_long = pd.concat(nrgrgd_all)

ax = sns.boxplot(x="Subject", y="ms", data=nrgrgd_all_long[nrgrgd_all_long['ms']!=0])
plt.title('Gaze Durations');
plt.show()

performance_answer = pd.Series(np.zeros(len(nrgrdur_all)))

for p,participant in enumerate(nrgrdur_all):
    answers = pd.read_csv((f"{base_dir}/{p+101}/{p+101}.txt"),
                            sep='\t',
                            header=0,
                            encoding='ANSI',
                            usecols=['Answer'])
    performance_answer[p] = len(answers[answers['Answer']=='Correct'])/40*100
    
description = pd.concat([n_regressions.astype(int),
                            n_blinks.astype(int),
                            n_trials_excluded.astype(int),
                            n_skipped_fp.astype(int),
                            percentage_good.round(2),
                            avg_ffd.round(2),
                            avg_gd.round(2),
                            performance_answer
                           ], axis=1)
description.columns = ['n_regressions',
                       'n_blinks',
                       'n_trials_excluded',
                       'n_skipped_firstpass',
                       'percentage_trials_kept',
                       'avg_ffd',
                       'avg_gd',
                       'comprehension']
    
wrgr_n_blinks = pd.Series(np.zeros(len(wrgrdur_all),dtype=int))
wrgr_n_regressions = pd.Series(np.zeros(len(wrgrdur_all),dtype=int))

wrgr_n_trials_excluded = pd.Series(np.zeros(len(wrgrffd_all)))
wrgr_n_skipped_fp = pd.Series(np.zeros(len(wrgrffd_all),dtype=int))
wrgr_n_goodtrials = pd.Series(np.zeros(len(wrgrffd_all),dtype=int))
wrgr_avg_ffd = pd.Series(np.zeros(len(wrgrffd_all)))
wrgr_avg_gd = pd.Series(np.zeros(len(wrgrgd_all)))


for p,participant in enumerate(wrgrdur_all):
    for trial in participant:
        if type(trial)==str:
            if trial == "Nope - there was regression":
                wrgr_n_regressions[p] += 1
            elif trial == "There was a blink":
                wrgr_n_blinks[p] += 1

for p,participant in enumerate(wrgrffd_all):
    wrgr_n_trials_excluded[p] = len(wrgrdur_all[p]) - len(wrgrffd_all[p])
    wrgr_n_skipped_fp[p] = len(participant[participant['ms']==0])
    wrgr_n_goodtrials[p] = participant['ms'][participant['ms']>0].count()/4
    wrgr_avg_ffd[p] = np.mean(participant['ms'][participant['ms']!=0])                
    
for p,participant in enumerate(wrgrgd_all):
    wrgr_avg_gd[p] = np.mean(participant['ms'][participant['ms']!=0])

for i,df, in enumerate(wrgrffd_all):
    wrgrffd_all[i]['Subject'] = [i]*len(wrgrffd_all[i])
wrgrffd_all_long = pd.concat(wrgrffd_all)    
    
ax = sns.boxplot(x="Subject", y="ms", data=wrgrffd_all_long[wrgrffd_all_long['ms']!=0])
plt.title('First Fixation Durations - with regressions');  
plt.show()  
    
for i,df, in enumerate(wrgrgd_all):
    wrgrgd_all[i]['Subject'] = [i]*len(wrgrgd_all[i])
wrgrgd_all_long = pd.concat(wrgrgd_all)

ax = sns.boxplot(x="Subject", y="ms", data=wrgrgd_all_long[wrgrgd_all_long['ms']!=0])
plt.title('Gaze Durations');
plt.show()

description_wrgr = pd.concat([wrgr_n_blinks.astype(int),
                            wrgr_n_trials_excluded.astype(int),
                            wrgr_n_skipped_fp.astype(int),
                            wrgr_n_goodtrials,
                            wrgr_avg_ffd.round(2),
                            wrgr_avg_gd.round(2),
                            performance_answer
                           ], axis=1)
description_wrgr.columns = ['n_blinks',
                       'n_trials_excluded',
                       'n_skipped_firstpass',
                       'percentage_trials_kept',
                       'avg_ffd',
                       'avg_gd',
                       'comprehension']

f, axes = plt.subplots(6, 5, figsize=(30,25))

for i in range(len(nrgrffd_all)):
    sns.regplot(data = nrgrffd_all_long[(nrgrffd_all_long['Subject']==i) & (nrgrffd_all_long['ms']>0)],
                x = 'Sim',
                y = 'ms',
                ax = axes[i//5,i%5])
    

f, axes = plt.subplots(6, 5, figsize=(30,25))

for i in range(len(wrgrffd_all)):
    sns.regplot(data = wrgrffd_all_long[(wrgrffd_all_long['Subject']==i) & (wrgrffd_all_long['ms']>0)],
                x = 'Sim',
                y = 'ms',
                ax = axes[i//5,i%5])    
    
f, axes = plt.subplots(6, 5, figsize=(30,25))

for i in range(len(nrgrffd_all)):
    sns.regplot(data = nrgrffd_all_long[(nrgrffd_all_long['Subject']==i) & (nrgrffd_all_long['ms']>0)],
                x = 'ConcM',
                y = 'ms',
                ax = axes[i//5,i%5])    
    
f, axes = plt.subplots(6, 5, figsize=(30,25))

for i in range(len(wrgrffd_all)):
    sns.regplot(data = wrgrffd_all_long[(wrgrffd_all_long['Subject']==i) & (wrgrffd_all_long['ms']>0)],
                x = 'ConcM',
                y = 'ms',
                ax = axes[i//5,i%5])    
    

    
    
    
