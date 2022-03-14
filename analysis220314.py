# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:40:29 2021

@author: fm02
"""

import os
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from pygazeanalyser.edfreader import read_edf

DISPSIZE = (1280, 1024)

# when talking about pixels, x,y=(0,0) is the top-left corner
#

# trials to exclude because of errors during recording
# they will be selected inside (normalised)_attach_info function
# (usually this means that the recording started before calibration)
# (key=subject_ID : values=trials_ID)
# consider that you should start counting trials from zero

# check records of test for comments, or also Demographic Info excel

exclude = {111: [120,121],
           128: [240,241],
           130: np.concatenate([[240],
                                 np.arange(120,133)]).tolist(),
           136: [80,81],
           141: np.arange(20,39).tolist()} 

 
def get_blinks(data_edf):
    blinks=[]
    for i,trial in enumerate(data_edf):
        blinks.append(data_edf[i]['events']['Eblk']) # get all blinks
        blinks = [x for x in blinks if x != []]
    blinks = [item for sublist in blinks for item in sublist]    
    return blinks 




def read_edf_plain(filename):
    """Get a dataframe containing only and all the events from the EDF file,
        with the trackertime, not dividing the trials"""
    # check if the file exists
    if os.path.isfile(filename):
		# open file
        f = open(filename, 'r')
	# raise exception if the file does not exist
    else:
        raise Exception("Error in read_edf: file '%s' does not exist" % filename)
    raw = f.readlines()
    f.close()
	# variables
    data = []
    event = []
    timepoint = []
	# loop through all lines
    for line in raw:
        if line[0:4] == "SFIX":
            l = line[9:]
            timepoint.append(int(l))
            event.append(line[0:4])
        elif line[0:4] == "EFIX":
            l = line[9:]
            l = l.split('\t')
            timepoint.append(int(l[1]))
            event.append(line[0:4])
         			# saccade start
        elif line[0:5] == 'SSACC':
            l = line[9:]
            timepoint.append(int(l))
            event.append(line[0:5])
         			# saccade end
        elif line[0:5] == "ESACC":
            l = line[9:]
            l = l.split('\t')
            timepoint.append(int(l[1]))
            event.append(line[0:5])
         			# blink start
        elif line[0:6] == "SBLINK":
            l = line[9:]
            timepoint.append(int(l))
            event.append(line[0:6])
         			# blink end
        elif line[0:6] == "EBLINK":
            l = line[9:]
            l = l.split('\t')
            timepoint.append(int(l[1]))
            event.append(line[0:6])
   	# return
    data = pd.DataFrame()
    data['time'] = np.array(timepoint)
    data['event'] = np.array(event)
    
    return data


def fixAOI(data_edf,data_plain):
    """Get all fixations within AOI. Checks that are not followed by a regression
    after the first fixation within the AOI + trials that do not contain
    a blink or error"""
    # get all fixation durations within a certain AOI for all trials for one subject
    # dur_all is the list where we include all the fixation durations that
    # respect certain inclusion criteria
    # 
    dur_all = []
    regressed = []
    time_before_fix = []
    tot_number_fixation = []
            
    for i,trial in enumerate(data_edf):
        pd_fix = pd.DataFrame.from_records(trial['events']['Efix'],
                                           columns=['start',
                                                    'end',
                                                    'duration',
                                                    'x',
                                                    'y'])
        tot_number_fixation.append(len(pd_fix))
        
        # exclude those trials where all fixations are outside the screen
        # it used to happe if there was an error in the gaze position detection
        # it should not be a problem now, considering that gaze is required
        # to trigger the start of the sentence
        if (((pd_fix['x'] < 0).all()) or ((pd_fix['x'] > DISPSIZE[0]).all())):
            dur_all.append('Error in fixation detection')
            time_before_fix.append(np.nan)
            regressed.append(np.nan)
        elif (((pd_fix['y'] < 0).all()) or ((pd_fix['y'] > DISPSIZE[1]).all())):
            dur_all.append('Error in fixation detection')
            time_before_fix.append(np.nan)
            regressed.append(np.nan)
        # or when no fixations have been detected
        elif len(pd_fix)<2:
            dur_all.append('Error in fixation detection')
            time_before_fix.append(np.nan)
            regressed.append(np.nan)
        else:
            # consider only fixations following a the first leftmost fixation

            # !! this is now useless and potentially problematic considering that
            # participants fixate on the left side of the screen (and not the centre)
            # before the appearance of the sentence
            
            # while pd_fix['x'][0]>pd_fix['x'][1]:
            #     pd_fix.drop([0],inplace=True)
            #     pd_fix.reset_index(drop=True, inplace=True)
            
            # the following info is gathered from the
            # stimulus presentation software (communicated the following msgs)
            
            # tuple indicating dimension of each sentence in pixels
            size = re.search("SIZE OF THE STIMULUS: (.*)\n",trial['events']['msg'][3][1])
            size = eval(size.group(1)) # tuple (width,height)
            
            # size of each letter in pixels
            # this should is identical for each sentence, equal to 11 in our study
            unit = re.search("NUMBER OF CHARACTERS: (.*)\n",trial['events']['msg'][4][1])
            unit = size[0]/eval(unit.group(1)) 
            
            # position (in characters) of the target word inside the sentence
            pos_target = re.search("POS TARGET INSIDE BOX: (.*)\n",trial['events']['msg'][5][1])
            pos_target = eval(pos_target.group(1))
            
            # position (in pixels) of the target word
            # convert width to the position in x, y cohordinates where the sentence starts
            # stimulus starting position is = centre of x_axis screen - half size of the sentence
            # because sentence is presented aligned to the centre of the screen
            pos_startstim = DISPSIZE[0]/2-size[0]/2
            # no need to calculate y as always in the same position at the centre
            # only one line
            
            # get x and y position of the target word
            # as pos_target is in characters, we need to mutiply each letter*unit
            # including in the AOI also half space preceding and half space
            # following the target word
            # tuple (x0,x1) position of the target word in pixels 
            target_x = (pos_startstim+(pos_target[0]*unit)-unit/2,pos_startstim+(pos_target[1]*unit)+unit/2)
            target_y = (DISPSIZE[1]/2-size[1]*2,DISPSIZE[1]/2+size[1]*2)
            # AOI for target_y position is two times the height of the letters
            # no need to be too strict as there's just one line
            
            # get all fixations on target word
            # this is checks if targetstart_position<fixation_position<targetend_position
            fixAOI = pd_fix['x'][(target_x[0]<pd_fix['x']) &
                                                  (pd_fix['x']<target_x[1]) &
                                                  (target_y[0]<pd_fix['y']) &
                                                  (pd_fix['y']<target_y[1])]
            
            # check if at least one fixation on target 
            if len(fixAOI)>0:
                
                # check this is first pass
                # by checking if all previous fixations (indetify by index) have a smaller x_position
                if all(pd_fix['x'][0:fixAOI.index[0]]<fixAOI[fixAOI.index[0]]):
                # check if this is not the last fixation
                    if (len(pd_fix['x'])>(fixAOI.index[0]+1)):
                        dur_all.append(pd_fix['duration'][(target_x[0]<pd_fix['x']) &
                                                  (pd_fix['x']<target_x[1]) &
                                                  (target_y[0]<pd_fix['y']) &
                                                  (pd_fix['y']<target_y[1])
                                                  ])
                        time_before_fix.append(pd_fix['start'][fixAOI.index[0]] - pd_fix['start'][0])
                    # check if there is a regression to BEFORE the target area
                    # and save it in the relevant list 
                        if (pd_fix['x'].iloc[fixAOI.index[0]+1]>target_x[0]):
                        # if there wasn't a regression, save as 0
                            regressed.append(0)
                        else:
                        # if there was a regression, save as 1
                            regressed.append(1)
                    else:
                    # if this is the last fixation, than there is no regression
                    # so, get the fixations (otherwise it will give an error
                    # when explicitly looking if fixation is followed by regression)
                    # however, there should always be a fixation after on the square
                        dur_all.append(pd_fix['duration'][(target_x[0]<pd_fix['x']) &
                                                  (pd_fix['x']<target_x[1]) &
                                                  (target_y[0]<pd_fix['y']) &
                                                  (pd_fix['y']<target_y[1])
                                                  ])
                        time_before_fix.append(pd_fix['start'][fixAOI.index[0]] - pd_fix['start'][0])
                        regressed.append(0)
                else:
                     dur_all.append('Nope - not fixated during first pass')       
                     time_before_fix.append(np.nan)
                     regressed.append(np.nan)
            else:
                # if there is no fixation, return empty Series
                dur_all.append(pd_fix['duration'][(target_x[0]<pd_fix['x']) &
                                                  (pd_fix['x']<target_x[1]) &
                                                  (target_y[0]<pd_fix['y']) &
                                                  (pd_fix['y']<target_y[1])
                                                  ])
                time_before_fix.append(np.nan)
                regressed.append(np.nan)
                
            # now check blinks
            # first, check at least one fixation and that the object is not string
            # remember that when trials are to be discarded there's a string
            if ((len(dur_all[-1])>0) and (type(dur_all[-1])!=str)):
                # get trackertime of the start of the first fixation on target words
                start = pd_fix['start'].iloc[dur_all[-1].index[0]]
                # get the  position in the events only list of data
                plain_start = data_plain[data_plain['time']==start].index[0]
                r = range(plain_start-2,plain_start+4)
                # this range because each blink generates an artefactual saccade
                # to each blink id surrounded by SSACC and ESAC events
                # different ends to include also EFIX event
                # this basically checks whether the fixation is immediately
                # preceded or followed by a blink
                if (any(data_plain['event'].iloc[r]=='SBLINK')
                    or any(data_plain['event'].iloc[r]=='EBLINK')):
                        dur_all[-1] = 'There was a blink'
                        time_before_fix[-1] = np.nan
                        regressed[-1] = np.nan
                        
    # returnign a list of series, containing all trials
    # each series contains all the fixations within AOI for that trial
    # each element consist in index = ordinal number of fixation for that trial
    # (eg if the first fixation within AOI was the 6th, index=6)
    # duration = duration of the fixation in ms
    return dur_all, regressed, time_before_fix, tot_number_fixation


# function to get both FFD and GD
def ffdgd(dur_all):
    """Get first-fixation, gaze duration, whether it was fixated"""
    # set everything to zero
    # this is convenient as words skipped have durations = 0 and fixated = 0
    FFD = np.zeros(len(dur_all))
    GD = np.zeros(len(dur_all))
    fixated = np.zeros(len(dur_all))
    n_prior_fixations = np.empty((len(dur_all)))
    n_prior_fixations[:] = np.nan
    for i,trial in enumerate(dur_all):
        # if error in fixation, then indicate as NAN
        if type(trial)==str: # all trials that should be excluded are strings ...
            if trial == 'Nope - not fixated during first pass':
                # ... apart if during first pass and  then fixated
                # note that in this case it counts as skipped (not as invalid!)
                pass
            else:
                # this will allow us to discard them from the analysis
                # when regressions, blinks or skipped on first pass
                FFD[i] = np.nan
                GD[i] = np.nan
                fixated[i] = np.nan
        else:
            # check if there is at least one fixation in AOI, otherwise FFD=GD=0                
            if len(dur_all[i])>0:
                # check if the fixation is btween 80-600ms long
                if (np.array(dur_all[i])[0]>80 and np.array(dur_all[i])[0]<600):
                    FFD[i] = np.array(dur_all[i])[0]
                    GD[i] = np.array(dur_all[i])[0]
                    fixated[i] = 1
                if len(dur_all[i])>1:
                    # if more than one, check whether they are consecutive
                    # fixations inside the AOI by checking the index
                    # see Footnote 1 
                    for j in range(len(dur_all[i].index)-1):
                        if ((dur_all[i].index[j+1]-dur_all[i].index[j]==1) &
                            (FFD[i]>0)):
                            GD[i] += np.array(dur_all[i])[j+1]
                        else:
                            break
                n_prior_fixations[i] = dur_all[i].index[0]
    return FFD, GD, fixated, n_prior_fixations
        
def attach_info(eyedata, regressed, time_before_ff, tot_number_fixation, n_prior_fix):
    """Include single word and sentence level statistics"""
    eyedata_all = []
    for i,participantdata in enumerate(eyedata):
        stimuli = pd.read_csv(f"{base_dir}/{participant[i]}/{participant[i]}.txt",
                              header=0,sep='\t',
                              encoding='ISO-8859-1')        
        eye_all_i = pd.DataFrame(list(zip(eyedata[i], stimuli.trialnr)),
                                 index=stimuli.IDstim,
                                 columns=['ms','trialnr'])
        eye_all_i['time_before_ff'] = time_before_ff[i]
        eye_all_i['regressed'] = regressed[i]
        eye_all_i['n_tot_fix'] = tot_number_fixation[i]
        eye_all_i['n_prior_fix'] = n_prior_fix[i]
        
        # check if need to exclude any trial
        if participant[i] in exclude:
            for tr_number in exclude[participant[i]]:
                eye_all_i.drop(eye_all_i[eye_all_i['trialnr']==tr_number].index,
                               inplace=True)
        
        # get predictors
        a = pd.merge(eye_all_i, stimuliALL[['ID',
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
                                     'Position',	
                                     'PRECEDING_Frequency',	
                                     'PRECEDING_LogFreq(Zipf)',	
                                     'LENprec',
                                     'cloze',
                                     'Sim',
                                     'plausibility',
                                     'SemD'
                                     ]], how='inner',left_on=['IDstim'],
                                         right_on=['ID'])
        eyedata_all.append(a)
        eyedata_all[-1] = eyedata_all[-1][eyedata_all[-1].iloc[:,0].notna()]    
    return eyedata_all

def attach_mean_centred(eyedata,regressed, time_before_ff, tot_number_fixation, n_prior_fix):
    """Supply the participants gd/ffd to obtain a gd/ffd_all that is mean_centred"""
    norm_eyedata_all = []
    for i,participantdata in enumerate(eyedata):
        stimuli = pd.read_csv(f"{base_dir}/{participant[i]}/{participant[i]}.txt",
                              header=0, sep='\t', encoding='ISO-8859-1')
        normalized_all_i = pd.DataFrame(list(zip(participantdata,
                                                 stimuli.trialnr)),
                                        index=stimuli.IDstim,
                                        columns=['ms','trialnr'])
        normalized_all_i['time_before_ff'] = time_before_ff[i]
        normalized_all_i['regressed'] = regressed[i]
        normalized_all_i['n_tot_fix'] = tot_number_fixation[i]
        normalized_all_i['n_prior_fix'] = n_prior_fix[i]
        normalized_all_i['time_before_ff'] = (normalized_all_i['time_before_ff'] - \
                                              normalized_all_i['time_before_ff'].mean() \
                                                  ) / normalized_all_i['time_before_ff'].std()
        normalized_all_i['n_tot_fix'] = (normalized_all_i['n_tot_fix'] - \
                                              normalized_all_i['n_tot_fix'].mean() \
                                                  ) / normalized_all_i['n_tot_fix'].std()
        normalized_all_i['n_prior_fix'] = (normalized_all_i['n_prior_fix'] - \
                                              normalized_all_i['n_prior_fix'].mean() \
                                                  ) / normalized_all_i['n_prior_fix'].std()            

            
        # check if need to exclude any trial
        if participant[i] in exclude:
            for tr_number in exclude[participant[i]]:
                normalized_all_i.drop(normalized_all_i[normalized_all_i['trialnr']==tr_number].index,
                               inplace=True)
        
        # get predictors
        normalized_all_i = pd.merge(normalized_all_i,
                                    stimuliALL_norm,
                                    how='inner',
                                    left_on=['IDstim'],
                                    right_on=['ID'])
        
        norm_eyedata_all.append(normalized_all_i)
        norm_eyedata_all[-1] = norm_eyedata_all[-1][norm_eyedata_all[-1].iloc[:,0].notna()]
    return norm_eyedata_all
        

####################################

DISPSIZE = (1280, 1024)
# Add information about target word
path = "C:/Users/fm02/OwnCloud/Sentences/"

os.chdir(path)
# stimuliALL = pd.read_excel('stimuli_all_onewordsemsim.xlsx', engine='openpyxl')
stimuliALL = pd.read_excel('stimuli_all_semD.xlsx', engine='openpyxl')

# include only numeric predictors
to_norm = stimuliALL[['ConcM','LEN','UN2_F','UN3_F','Orth','OLD20','FreqCount','LogFreq(Zipf)', 
                     'V_MeanSum','A_MeanSum','mink3_SM','BLP_rt','BLP_accuracy',
                     'similarity','Position','PRECEDING_Frequency','PRECEDING_LogFreq(Zipf)',	
                     'LENprec','Predictability','cloze','plausibility','Sim','SemD']]
to_norm = (to_norm-to_norm.mean())/to_norm.std()
# put back Word and ID
stimuliALL_norm = stimuliALL[['Word','ID']].join(to_norm)

# import data from the participants
base_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/"

participant = [
        101, 
        102, 
        103, 
        104, 
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
#        139 # excluded - not completed testing
        140,
        141
        ]

data = {}
data_plain = {}

for i in participant:
    print(f'Reading EDF data participant {i}')
    data[i] = read_edf(f"{base_dir}/{i}/{i}.asc",
                       "STIMONSET","STIMOFFSET")
    data_plain[i] = read_edf_plain(f"{base_dir}/{i}/{i}.asc")

# prefix nrgr = no_regressions
dur = []
regressed = []
time_before = []
nfix = []

ffd = []
gd = []
prfix = []
nprior_fixs = []


# loop over participants  
for subject in data.keys():
    print(f'Extracting data participant {subject}')
    dur_i, regressed_i, time_before_i, nfix_i = fixAOI(data[subject],
                                                        data_plain[subject])
    
    FFD_i, GD_i, fixated_i, nprior_fixs_i = ffdgd(dur_i)
    
    dur.append(dur_i)
    regressed.append(regressed_i)
    time_before.append(time_before_i)
    nfix.append(nfix_i)
    
    ffd.append(FFD_i)
    gd.append(GD_i)
    prfix.append(fixated_i)
    nprior_fixs.append(nprior_fixs_i)    

gd_all = attach_info(gd, regressed, time_before, nfix, nprior_fixs)
ffd_all = attach_info(ffd, regressed, time_before, nfix, nprior_fixs)
norm_gd_all = attach_mean_centred(gd, regressed, time_before, nfix, nprior_fixs)
norm_ffd_all = attach_mean_centred(ffd, regressed, time_before, nfix, nprior_fixs)

pis = pd.read_excel("//cbsu/data/Imaging/hauk/users/fm02/EOS_data/Demographic_info.xlsx",
                    usecols=["Participant ID",
                             "Gender",	
                             "Age",	
                             "Handedness",	
                             "% Correct Responses"])

################ need to run this through all participants and save ########


#####################
### SAVE ############
#####################

for i,df in enumerate(norm_ffd_all):
    norm_ffd_all[i] = norm_ffd_all[i].rename(columns={'LogFreq(Zipf)':'LogFreqZipf',
                                                    'PRECEDING_LogFreq(Zipf)':'PRECEDING_LogFreqZipf'})
    norm_ffd_all[i]['Subject'] = [i]*len(norm_ffd_all[i])
    norm_ffd_all[i]['Gender'] = [pis["Gender"][pis["Participant ID"] == participant[i]].values[0]] \
                                        *len(norm_ffd_all[i])
    norm_ffd_all[i]['Age'] = [pis["Age"][pis["Participant ID"] == participant[i]].values[0]] \
                                        *len(norm_ffd_all[i])
      
# GD  - no regressions and normalised predictors, which is probably what we will use   

for i,df, in enumerate(norm_gd_all):
    norm_gd_all[i] = norm_gd_all[i].rename(columns={'LogFreq(Zipf)':'LogFreqZipf',
                                                    'PRECEDING_LogFreq(Zipf)':'PRECEDING_LogFreqZipf'})
    norm_gd_all[i]['Subject'] = [i]*len(norm_gd_all[i])
    norm_gd_all[i]['Gender'] = [pis["Gender"][pis["Participant ID"] == participant[i]].values[0]] \
                                        *len(norm_gd_all[i])
    norm_gd_all[i]['Age'] = [pis["Age"][pis["Participant ID"] == participant[i]].values[0]] \
                                        *len(norm_gd_all[i])
      

for dat,name in zip([dur,
            ffd_all,
            gd_all,
            norm_ffd_all,
            norm_gd_all],
            ['dur_all',
            'ffd_all',
            'gd_all',
            'norm_ffd_all',
            'norm_gd_all']):
    participants = {}
    for i,df in enumerate(dat):
        participants[i] = df
    
    with open(f"U:/AnEyeOnSemantics/41analysis/{name}_withSemD.P", 'wb') as outfile:
        pickle.dump(participants,outfile)  

pd.concat(norm_ffd_all).to_csv('C:/Users/fm02/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_ffd_41_withSemD.csv',index=False)

pd.concat(norm_gd_all).to_csv('C:/Users/fm02/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_gd_41_withSemD.csv',index=False)

##################
### Footnote 1 ###
### In this way, if a first fixation within the AOI is less then 80ms,
### but a participant performs a subsequent fixation within the AOI, the two
### consecutive fixatinos will be summed in the calculation of GD, but count as
### zero in the FFD calculation. This affects only 14 trials :
# ### ID Subject
# 1  291       1
# 2  444       1
# 3  255       1
# 4  354      10
# 5  304      10
# 6  198      10
# 7  487      17
# 8  309      17
# 9  358      17
# 10 303      20
# 11 244      20
# 12 123      20
# 13 417      28
# 14 149      39
### Are there better ways to deal with this? e.g., consider the sum of them
### also when calculating FFD? -> i.e., do them count as one fixation, two
### consecutive fixations or half way?