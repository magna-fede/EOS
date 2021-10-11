# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:01:09 2021

@author: magna.fede
"""
### This script gets the output from the EyeLink EDF file (converted to asc)
### and analyses it.
### The output consists in files which will be used in R to create LMMs 
### for statistical analysis.
### Consider that fixations are determined by EyeLink's algortithm, 
### we are just determining which fixations are
### inside the AOI= fixations on the target words.

### This should be run in Spyder, especially for visualization purposes.
### If only want to generate the data for R, this (should) work as a script,
### be sure to uncomment the saving section.

# import relevant stuff
import os
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

from pygazeanalyser.edfreader import read_edf

DISPSIZE = (1280, 1024)

# when talking about pixels, x,y=(0,0) is the top-left corner
# 
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


def no_lookback_firstAOI(data_edf,data_plain):
    """Get all fixations within AOI that are not followed by a regression
    after the first fixation within the AOI + trials that do not contain
    a blink"""
    # get all fixation durations within a certain AOI for all trials for one subject
    # dur_all is the list where we include all the fixation durations that
    # respect certain inclusion criteria
    # 
    dur_all = []
    for i,trial in enumerate(data_edf):
        pd_fix = pd.DataFrame.from_records(trial['events']['Efix'],
                                           columns=['start',
                                                    'end',
                                                    'duration',
                                                    'x',
                                                    'y'])
        
        # exclude those trials where all fixations are outside the screen
        # it used to happe if there was an error in the gaze position detection
        # it should not be a problem now, considering that gaze is required
        # to trigger the start of the sentence
        if (((pd_fix['x'] < 0).all()) or ((pd_fix['x'] > DISPSIZE[0]).all())):
            dur_all.append('Error in fixation detection')  
        elif (((pd_fix['y'] < 0).all()) or ((pd_fix['y'] > DISPSIZE[1]).all())):
            dur_all.append('Error in fixation detection')
        
        # or when no fixations have been detected
        elif len(pd_fix)<2:
            dur_all.append('Error in fixation detection')
            
        else:
            # consider only fixations following a the first leftmost fixation
            while pd_fix['x'][0]>pd_fix['x'][1]:
                pd_fix.drop([0],inplace=True)
                pd_fix.reset_index(drop=True, inplace=True)
            
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
                    # check if there is a regression to BEFORE the target area
                        if (pd_fix['x'].iloc[fixAOI.index[0]+1]>target_x[0]):
                        # if not, get duration
                            dur_all.append(pd_fix['duration'][(target_x[0]<pd_fix['x']) &
                                                  (pd_fix['x']<target_x[1]) &
                                                  (target_y[0]<pd_fix['y']) &
                                                  (pd_fix['y']<target_y[1])
                                                  ])
                        else:
                        # if there was a regression, do not consider the trial
                            dur_all.append('Nope - there was regression')
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
                else:
                     dur_all.append('Nope - not fixated during first pass')       
            else:
                # if there is no fixation, return empty Series
                dur_all.append(pd_fix['duration'][(target_x[0]<pd_fix['x']) &
                                                  (pd_fix['x']<target_x[1]) &
                                                  (target_y[0]<pd_fix['y']) &
                                                  (pd_fix['y']<target_y[1])
                                                  ])
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
                        
    # returnign a list of series, containing all trials
    # each series contains all the fixations within AOI for that trial
    # each element consist in index = ordinal number of fixation for that trial
    # (eg if the first fixation within AOI was the 6th, index=6)
    # duration = duration of the fixation in ms
    return dur_all

# same as before, but allowing regressions
# not commented as before, but doing the same things, just with one check less
def allowing_lookback_firstAOI(data_edf,data_plain):
    """Get all fixations within AOI that are not followed by a regression
    after the first fixation within the AOI + trials that do not contain
    a blink"""
    # get all fixation durations within a certain AOI for all trials for one subject
    # dur_all is the list where we include all the fixation durations that
    # respect certain inclusion criteria
    # 
    dur_all = []
    for i,trial in enumerate(data_edf):
        # get only 'Efix' events, because we are interested in fixations and this is where info is sotered
        pd_fix = pd.DataFrame.from_records(trial['events']['Efix'],
                                           columns=['start',
                                                    'end',
                                                    'duration',
                                                    'x',
                                                    'y'])
        
        # exclude those trials where all fixations are outside the screen
        # can happen if there is an error in the gaze position detection
        # it should not be a problem now, considering that gaze is required
        # to trigger the start of the sentence
        if (((pd_fix['x'] < 0).all()) or ((pd_fix['x'] > DISPSIZE[0]).all())):
            dur_all.append('Error in fixation detection')  
        elif (((pd_fix['y'] < 0).all()) or ((pd_fix['y'] > DISPSIZE[1]).all())):
            dur_all.append('Error in fixation detection')
        
        # or when no fixations have been detected
        elif len(pd_fix)<2:
            dur_all.append('Error in fixation detection')
            
        else:
            # consider only fixations following a the first leftmost fixation
            while pd_fix['x'][0]>pd_fix['x'][1]:
                pd_fix.drop([0],inplace=True)
                pd_fix.reset_index(drop=True, inplace=True)
            
            # the following info is specifc to our script, where from the
            # stimulus presentation software we communicate the following msgs
            
            # tuple indicating dimension of each sentence in pixels
            # this is the result of a boundingBox around the stimulus
            # please, check PsychoPy documentation if unsure
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
            pos_startstim = DISPSIZE[0]/2-size[0]/2
            # no need to calculate y as always in the same position at the centre
            # only one line
            # CAREFUL! this assumes that the sentence is aligned to the center
            
            # get x and y position of the target word
            # as pos_target is in characters, we need to mutiply each letter*unit
            # including in the AOI also half space preceding and half space
            # following the target word
            # tuple (x0,x1) position of the target word in pixels 
            target_x = (pos_startstim+(pos_target[0]*unit)-unit/2,pos_startstim+(pos_target[1]*unit)+unit/2)
            target_y = (DISPSIZE[1]/2-size[1]*2,DISPSIZE[1]/2+size[1]*2)
            # AOI for target_y position is two times the height of the letters
            # no need to be too strict as there's just one line
            
            # get all fixations on target x position
            fixAOI = pd_fix['x'][(target_x[0]<pd_fix['x']) &
                                                  (pd_fix['x']<target_x[1]) &
                                                  (target_y[0]<pd_fix['y']) &
                                                  (pd_fix['y']<target_y[1])]
            # check if at least one fixation
            if len(fixAOI)>0:    
                # check this is first pass
                if all(pd_fix['x'][0:fixAOI.index[0]]<fixAOI[fixAOI.index[0]]):
                    dur_all.append(pd_fix['duration'][(target_x[0]<pd_fix['x']) &
                                                  (pd_fix['x']<target_x[1]) &
                                                  (target_y[0]<pd_fix['y']) &
                                                  (pd_fix['y']<target_y[1])
                                                  ])
                else:
                     dur_all.append('Nope - not fixated during first pass')       
            else:
                # if there is no fixation, return empty Series
                dur_all.append(pd_fix['duration'][(target_x[0]<pd_fix['x']) &
                                                  (pd_fix['x']<target_x[1]) &
                                                  (target_y[0]<pd_fix['y']) &
                                                  (pd_fix['y']<target_y[1])
                                                  ])
    
        
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
                        
    # returnign a list of series, containing all trials
    # each series contains all the fixations within AOI for that trial
    # each element consist in index = ordinal number of fixation for that trial
    # (eg if the first fixation within AOI was the 6th, index=6)
    # duration = duration of the fixation in ms
    return dur_all



# function to get both FFD and GD
def ffdgd(dur_all):
    """Get first-fixation, gaze duration, whether it was fixated"""
    # set everything to zero
    # this is convenient as words skipped have durations = 0 and fixated = 0
    FFD = np.zeros(len(dur_all))
    GD = np.zeros(len(dur_all))
    fixated = np.zeros(len(dur_all))
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
                    for j in range(len(dur_all[i].index)-1):
                        if dur_all[i].index[j+1]-dur_all[i].index[j]==1:
                            GD[i] += np.array(dur_all[i])[j+1]
                        else:
                            break
    return FFD, GD, fixated
        
def attach_info(eyedata):
    """Include single word and sentence level statistics"""
    eyedata_all = []
    for i,participantdata in enumerate(eyedata):
        stimuli = pd.read_csv(data_dir[i]+'.txt', header=0,sep='\t',encoding='ISO-8859-1')
        eye_all_i = pd.DataFrame(list(zip(eyedata[i],stimuli.trialnr)), index=stimuli.IDstim, columns=['ms','trialnr'])
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
                                     'plausibility'
                                     ]], how='inner',left_on=['IDstim'], right_on=['ID'])
        eyedata_all.append(a)
        eyedata_all[-1] = eyedata_all[-1][eyedata_all[-1].iloc[:,0].notna()]    
    return eyedata_all

def attach_mean_centred(eyedata):
    """Supply the participants gd/ffd to obtain a gd/ffd_all that is mean_centred"""
    norm_eyedata_all = []
    for i,participantdata in enumerate(eyedata):
        stimuli = pd.read_csv(data_dir[i]+'.txt', header=0,sep='\t',encoding='ISO-8859-1')
        normalized_all_i = pd.DataFrame(list(zip(participantdata,stimuli.trialnr)), index=stimuli.IDstim, columns=['ms','trialnr'])
        # get predictors
        normalized_all_i = pd.merge(normalized_all_i, stimuliALL_norm, how='inner',left_on=['IDstim'], right_on=['ID'])
        norm_eyedata_all.append(normalized_all_i)
        norm_eyedata_all[-1] = norm_eyedata_all[-1][norm_eyedata_all[-1].iloc[:,0].notna()]
    return norm_eyedata_all

############THIS is just an idea on how to include the number of fixations prior the fixation of interest
# # get the number of fixations prior to the first fixation within AOI
# n_firstfix = []
# for i,trial in enumerate(dur3):
#     if len(trial) > 0:
#         if type(trial)!=str:
#             n_firstfix.append(dur3[i].index[0])

DISPSIZE = (1280, 1024)
# Add information about target word
path = "C:/Users/fm02/OwnCloud/Sentences/"

os.chdir(path)
stimuliALL = pd.read_excel('stimuli_all_onewordsemsim.xlsx', engine='openpyxl')
# include only numeric predictors
to_norm = stimuliALL[['ConcM','LEN','UN2_F','UN3_F','Orth','OLD20','FreqCount','LogFreq(Zipf)', 
                     'V_MeanSum','A_MeanSum','mink3_SM','BLP_rt','BLP_accuracy',
                     'similarity','Position','PRECEDING_Frequency','PRECEDING_LogFreq(Zipf)',	
                     'LENprec','Predictability','cloze','plausibility','Sim']]
to_norm = (to_norm-to_norm.mean())/to_norm.std()
# put back Word and ID
stimuliALL_norm = stimuliALL[['Word','ID']].join(to_norm)

# import data from the participants
data101_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/101/101"

data_dir = [data101_dir]

# this is pygaze standard read_edf function
data101 = read_edf(data101_dir+".asc","STIMONSET","STIMOFFSET")

# this is a modified version of the above, which just saves the events
# and does not separate the trials
# we'll use it to determine trials that contain blinks 

data101_plain = read_edf_plain(data101_dir+".asc")

data = [data101]
data_plain = [data101_plain]

# prefix nrgr = no_regressions
nrgrdur = []

# prefix wrgr = accepting regressions
wrgrdur = []

nrgrffd = []
nrgrgd = []
nrgrprfix = []

wrgrffd = []
wrgrgd = []
wrgrprfix = []
    
# loop over participants  
for i,dat in enumerate(data):
    nrgrdur_i = no_lookback_firstAOI(data[i],data_plain[i])
    wrgrdur_i = allowing_lookback_firstAOI(data[i],data_plain[i])
    
    nrgrFFD_i, nrgrGD_i, nrgrfixated_i = ffdgd(nrgrdur_i)
    wrgrFFD_i, wrgrGD_i, wrgrfixated_i = ffdgd(wrgrdur_i)
    
    nrgrdur.append(nrgrdur_i)
    nrgrffd.append(nrgrFFD_i)
    nrgrgd.append(nrgrGD_i)
    nrgrprfix.append(nrgrfixated_i)
    
    wrgrdur.append(wrgrdur_i)
    wrgrffd.append(wrgrFFD_i)
    wrgrgd.append(wrgrGD_i)
    wrgrprfix.append(wrgrfixated_i)

# attach for convenience the single-word and sentence-level statistics
nrgrgd_all = attach_info(nrgrgd)
nrgrffd_all = attach_info(nrgrffd)
nrgrfixations_all = attach_info(nrgrprfix)

wrgrgd_all = attach_info(wrgrgd)
wrgrffd_all = attach_info(wrgrffd)
wrgrfixations_all = attach_info(wrgrprfix)


norm_nrgrgd_all = attach_mean_centred(nrgrgd)
norm_nrgrffd_all = attach_mean_centred(nrgrffd)
norm_nrgrfixations_all = attach_mean_centred(nrgrprfix)

norm_wrgrgd_all = attach_mean_centred(wrgrgd)
norm_wrgrffd_all = attach_mean_centred(wrgrffd)
norm_wrgrfixations_all = attach_mean_centred(wrgrprfix)

##################################################################################################################
##################################################################################################################
##################################################################################################################

# ### HERE ARE PLOTS, uncomment if necessary
# ###### visualization purposes only, right now only one subject (will create code for average)

# ax = sns.regplot(data=nrgrffd_all[0][nrgrffd_all[0]['ms']>0], x='Sim',y='ms')
# ax.set_title('First Fixation duration - Cloze SemanticSimilarity', fontsize = 15);

# bx = sns.regplot(data=nrgrffd_all[0][nrgrffd_all[0]['ms']>0], x='cloze',y='ms')
# bx.set_title('First Fixation duration - Cloze', fontsize = 15);

# cx = sns.regplot(data=nrgrffd_all[0][nrgrffd_all[0]['ms']>0], x='LogFreq(Zipf)',y='ms')
# cx.set_title('First Fixation duration - LogFrequency (Zipf)', fontsize = 15);

# dx = sns.regplot(data=norm_nrgrgd_all[0][norm_nrgrgd_all[0]['ms']>0], x='ConcM',y='ms')
# dx.set_title('Gaze duration - Concreteness', fontsize = 15);

# ########### WORK IN PROGRESS ##############
# # ax = sns.regplot(data=Nffd_all[3], x='Predictability',y='ms')
 

# def new_concM(df_cm):
#     if df_cm<=2:
#         a = 1
#     elif (2<df_cm<=3):
#         a = 2
#     elif (3<df_cm<=4):
#         a = 3
#     elif (4<df_cm<=5):
#         a = 4
#     return a

# def new_clz(df_cl):
#     if df_cl<=0.15:
#         a = 0

#     else:
#         a = 1
#     return a  

# wrgrffd_all[1]['nCM'] = wrgrffd_all[1]['ConcM'].apply(new_concM)
# wrgrffd_all[1]['nCL'] = wrgrffd_all[1]['cloze'].apply(new_clz)


# plt.subplot()
# sns.lmplot(x="Sim", y="ms",
#            hue="nCM",data=wrgrffd_all[1][wrgrffd_all[1]['ms']>0],
#            palette="viridis")
# plt.legend(['abstract','moderately abstract', 'moderately concrete', 'concrete'])

# nrgrffd_all[1]['nCM'] = nrgrffd_all[1]['ConcM'].apply(new_concM)
# nrgrffd_all[1]['nCL'] = nrgrffd_all[1]['cloze'].apply(new_clz)


# plt.subplot()
# sns.lmplot(x="Sim", y="ms",
#            hue="nCM",data=nrgrffd_all[1][nrgrffd_all[1]['ms']>0],
#            palette="viridis")
# plt.legend(['abstract','moderately abstract', 'moderately concrete', 'concrete'])


# plt.subplot()
# sns.lmplot(x='ConcM', y="ms",
#            hue="nCL",data=wrgrffd_all[2][wrgrffd_all[2]['ms']>0],
#            palette="viridis")

##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
# HERE we save the files which will be used for statistical analysis in R

# this corrects the name of one of the predictors, or R will complain about the parenthesis
# and adds a column with subject identity (which will be needed in the model)
# FFD - no regression and normalised predictors, which is probably what we will use
for i,df, in enumerate(norm_nrgrffd_all):
    norm_nrgrffd_all[i] = norm_nrgrffd_all[i].rename(columns={'LogFreq(Zipf)':'LogFreqZipf',
                                                    'PRECEDING_LogFreq(Zipf)':'PRECEDING_LogFreqZipf'})
    norm_nrgrffd_all[i]['Subject'] = [i]*len(norm_nrgrffd_all[i])
# GD  - no regressions and normalised predictors, which is probably what we will use   
for i,df, in enumerate(norm_nrgrgd_all):
    norm_nrgrgd_all[i] = norm_nrgrgd_all[i].rename(columns={'LogFreq(Zipf)':'LogFreqZipf',
                                                    'PRECEDING_LogFreq(Zipf)':'PRECEDING_LogFreqZipf'})
    norm_nrgrgd_all[i]['Subject'] = [i]*len(norm_nrgrgd_all[i])
# #  the following are alternative analysis, which probably will not be used   
# # FFD no regression raw predictors
# for i,df, in enumerate(nrgrffd_all):
#     nrgrffd_all[i] = nrgrffd_all[i].rename(columns={'LogFreq(Zipf)':'LogFreqZipf',
#                                                     'PRECEDING_LogFreq(Zipf)':'PRECEDING_LogFreqZipf'})
#     nrgrffd_all[i]['Subject'] = [i]*len(nrgrffd_all[i])
# # GD no regressions raw predictors
# for i,df, in enumerate(norm_wrgrffd_all):
#     norm_wrgrffd_all[i] = norm_wrgrffd_all[i].rename(columns={'LogFreq(Zipf)':'LogFreqZipf',
#                                                     'PRECEDING_LogFreq(Zipf)':'PRECEDING_LogFreqZipf'})
#     norm_wrgrffd_all[i]['Subject'] = [i]*len(norm_wrgrffd_all[i])
# # GD with regressions normalised    
# for i,df, in enumerate(norm_wrgrgd_all):
#     norm_wrgrgd_all[i] = norm_wrgrgd_all[i].rename(columns={'LogFreq(Zipf)':'LogFreqZipf',
#                                                     'PRECEDING_LogFreq(Zipf)':'PRECEDING_LogFreqZipf'})
#     norm_wrgrgd_all[i]['Subject'] = [i]*len(norm_wrgrgd_all[i])
# # FFD with regression raw predictors    
# for i,df, in enumerate(wrgrffd_all):
#     wrgrffd_all[i] = wrgrffd_all[i].rename(columns={'LogFreq(Zipf)':'LogFreqZipf',
#                                                     'PRECEDING_LogFreq(Zipf)':'PRECEDING_LogFreqZipf'})
#     wrgrffd_all[i]['Subject'] = [i]*len(wrgrffd_all[i])


# this saves to a csv file the data which will be used in the analysis
# change path as necessary
pd.concat(norm_nrgrffd_all).to_csv('C:/Users/User/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_ffdpilots_onesemsim0924.csv',index=False)

pd.concat(norm_nrgrgd_all).to_csv('C:/Users/User/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_gdpilots_onesemsim0924.csv',index=False)

# pd.concat(nrgrffd_all).to_csv('C:/Users/User/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/ffdpilots_onesemsim.csv',index=False)

# pd.concat(norm_wrgrffd_all).to_csv('C:/Users/User/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_ffdpilots_withregression_onesemsim.csv',index=False)

# pd.concat(norm_wrgrgd_all).to_csv('C:/Users/User/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_gdpilots_withregression_onesemsim.csv',index=False)

# pd.concat(wrgrffd_all).to_csv('C:/Users/User/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/ffdpilots_withregression_onesemsim.csv',index=False)
