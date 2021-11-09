# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:01:0[9] 2021

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

# trials to exclude because of errors during recording
# they will be selected inside (normalised)_attach_info function
# (usually this means that the recording started before calibration)
# (key=subject_ID : values=trials_ID)
# consider that you should start counting trials from zero

exclude = {111:[120,121]} 



 
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
    
    time_before_fix = []
            
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
            time_before_fix.append(0)
        elif (((pd_fix['y'] < 0).all()) or ((pd_fix['y'] > DISPSIZE[1]).all())):
            dur_all.append('Error in fixation detection')
            time_before_fix.append(0)
        # or when no fixations have been detected
        elif len(pd_fix)<2:
            dur_all.append('Error in fixation detection')
            time_before_fix.append(0)
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
                    # check if there is a regression to BEFORE the target area
                        if (pd_fix['x'].iloc[fixAOI.index[0]+1]>target_x[0]):
                        # if not, get duration
                            dur_all.append(pd_fix['duration'][(target_x[0]<pd_fix['x']) &
                                                  (pd_fix['x']<target_x[1]) &
                                                  (target_y[0]<pd_fix['y']) &
                                                  (pd_fix['y']<target_y[1])
                                                  ])
                            time_before_fix.append(pd_fix['start'][fixAOI.index[0]] - pd_fix['start'][0])
                        else:
                        # if there was a regression, do not consider the trial
                            dur_all.append('Nope - there was regression')
                            time_before_fix.append(pd_fix['start'][fixAOI.index[0]] - pd_fix['start'][0])

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
                else:
                     dur_all.append('Nope - not fixated during first pass')       
                     time_before_fix.append(pd_fix['start'][fixAOI.index[0]] - pd_fix['start'][0])
            else:
                # if there is no fixation, return empty Series
                dur_all.append(pd_fix['duration'][(target_x[0]<pd_fix['x']) &
                                                  (pd_fix['x']<target_x[1]) &
                                                  (target_y[0]<pd_fix['y']) &
                                                  (pd_fix['y']<target_y[1])
                                                  ])
                time_before_fix.append(np.nan)
                
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
                        time_before_fix[-1] = 0
                        
    # returnign a list of series, containing all trials
    # each series contains all the fixations within AOI for that trial
    # each element consist in index = ordinal number of fixation for that trial
    # (eg if the first fixation within AOI was the 6th, index=6)
    # duration = duration of the fixation in ms
    return dur_all, time_before_fix

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
    time_before_fix = []
    
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
            time_before_fix.append(0)
        elif (((pd_fix['y'] < 0).all()) or ((pd_fix['y'] > DISPSIZE[1]).all())):
            dur_all.append('Error in fixation detection')
            time_before_fix.append(0)
        
        # or when no fixations have been detected
        elif len(pd_fix)<2:
            dur_all.append('Error in fixation detection')
            time_before_fix.append(0)
            
        else:
            # consider only fixations following a the first leftmost fixation
            # while pd_fix['x'][0]>pd_fix['x'][1]:
            #     pd_fix.drop([0],inplace=True)
            #     pd_fix.reset_index(drop=True, inplace=True)
            
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
                    time_before_fix.append(pd_fix['start'][fixAOI.index[0]] - pd_fix['start'][0])

                else:
                     dur_all.append('Nope - not fixated during first pass')       
                     time_before_fix.append(pd_fix['start'][fixAOI.index[0]] - pd_fix['start'][0])

            else:
                # if there is no fixation, return empty Series
                dur_all.append(pd_fix['duration'][(target_x[0]<pd_fix['x']) &
                                                  (pd_fix['x']<target_x[1]) &
                                                  (target_y[0]<pd_fix['y']) &
                                                  (pd_fix['y']<target_y[1])
                                                  ])
                time_before_fix.append(np.nan)
        
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
                        time_before_fix[-1] = 0
                        
    # returnign a list of series, containing all trials
    # each series contains all the fixations within AOI for that trial
    # each element consist in index = ordinal number of fixation for that trial
    # (eg if the first fixation within AOI was the 6th, index=6)
    # duration = duration of the fixation in ms
    return dur_all, time_before_fix



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
        
def attach_info(eyedata, time_before_ff):
    """Include single word and sentence level statistics"""
    eyedata_all = []
    for i,participantdata in enumerate(eyedata):
        stimuli = pd.read_csv(data_dir[i]+'.txt', header=0,sep='\t',
                              encoding='ISO-8859-1')        
        eye_all_i = pd.DataFrame(list(zip(eyedata[i], stimuli.trialnr)),
                                 index=stimuli.IDstim,
                                 columns=['ms','trialnr'])
        eye_all_i['time_before_ff'] = time_before_ff[i]
        
        # check if need to exclude any trial
        if i+101 in exclude:
            for tr_number in exclude[i+101]:
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
                                     'plausibility'
                                     ]], how='inner',left_on=['IDstim'],
                                         right_on=['ID'])
        eyedata_all.append(a)
        eyedata_all[-1] = eyedata_all[-1][eyedata_all[-1].iloc[:,0].notna()]    
    return eyedata_all

def attach_mean_centred(eyedata, time_before_ff):
    """Supply the participants gd/ffd to obtain a gd/ffd_all that is mean_centred"""
    norm_eyedata_all = []
    for i,participantdata in enumerate(eyedata):
        stimuli = pd.read_csv(data_dir[i]+'.txt',
                              header=0, sep='\t', encoding='ISO-8859-1')
        normalized_all_i = pd.DataFrame(list(zip(participantdata,
                                                 stimuli.trialnr)),
                                        index=stimuli.IDstim,
                                        columns=['ms','trialnr'])
        normalized_all_i['time_before_ff'] = time_before_ff[i]
        normalized_all_i['time_before_ff'] = (normalized_all_i['time_before_ff'] - \
                                              normalized_all_i['time_before_ff'].mean() \
                                                  ) / normalized_all_i['time_before_ff'].std()
        
        # check if need to exclude any trial
        if i+101 in exclude:
            for tr_number in exclude[i+101]:
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

def count_events(eyedata):
    """
    Supply the participants dur to obtain a dataframe containing description
    of the events that happened for that participant. For example:
        - errors during recording
        - number of regressions
        - probability of skipping the target word
        - average fixation duration
        - average gaze duration
    """

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
data102_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/102/102"
data103_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/103/103"
data104_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/104/104"
data105_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/105/105"
data106_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/106/106"
data107_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/107/107"
data108_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/108/108"
data109_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/109/109"
data110_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/110/110"
data111_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/111/111"
data112_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/112/112"
data113_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/113/113"
data114_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/114/114"
data115_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/115/115"
data116_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/116/116"
data117_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/117/117"
# data118_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/118/118"
# data119_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/119/119"
# data120_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/120/120"
# data121_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/121/121"
# data122_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/122/122"
# data123_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/123/123"
# data124_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/124/124"
# data125_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/125/125"
# data126_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/126/126"
# data127_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/127/127"
# data128_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/128/128"
# data129_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/129/129"
# data130_dir = "//cbsu/data/Imaging/hauk/users/fm02/EOS_data/data_fromLab/130/130"


data_dir = [data101_dir, 
            data102_dir, 
            data103_dir, 
            data104_dir, 
            data105_dir,
            data106_dir,
            data107_dir,
            data108_dir,
            data109_dir,
            data110_dir,
            data111_dir,
            data112_dir,
            data113_dir,
            data114_dir,
            data115_dir,
            data116_dir,
            data117_dir,
            # data118_dir,
              # data119_dir,
              # data120_dir,
              # data121_dir,
              # data122_dir,
              # data123_dir,
              # data124_dir,
              # data125_dir,
              # data126_dir,
              # data127_dir,
              # data128_dir,
              # data129_dir,
              # data130_dir
            ]

# this is pygaze standard read_edf function
data101 = read_edf(data101_dir+".asc","STIMONSET","STIMOFFSET")
data102 = read_edf(data102_dir+".asc","STIMONSET","STIMOFFSET")
data103 = read_edf(data103_dir+".asc","STIMONSET","STIMOFFSET")
data104 = read_edf(data104_dir+".asc","STIMONSET","STIMOFFSET")
data105 = read_edf(data105_dir+".asc","STIMONSET","STIMOFFSET")
data106 = read_edf(data106_dir+".asc","STIMONSET","STIMOFFSET")
data107 = read_edf(data107_dir+".asc","STIMONSET","STIMOFFSET")
data108 = read_edf(data108_dir+".asc","STIMONSET","STIMOFFSET")
data109 = read_edf(data109_dir+".asc","STIMONSET","STIMOFFSET")
data110 = read_edf(data110_dir+".asc","STIMONSET","STIMOFFSET")
data111 = read_edf(data111_dir+".asc","STIMONSET","STIMOFFSET")
data112 = read_edf(data112_dir+".asc","STIMONSET","STIMOFFSET")
data113 = read_edf(data113_dir+".asc","STIMONSET","STIMOFFSET")
data114 = read_edf(data114_dir+".asc","STIMONSET","STIMOFFSET")
data115 = read_edf(data115_dir+".asc","STIMONSET","STIMOFFSET")
data116 = read_edf(data116_dir+".asc","STIMONSET","STIMOFFSET")
data117 = read_edf(data117_dir+".asc","STIMONSET","STIMOFFSET")
# data118 = read_edf(data118_dir+".asc","STIMONSET","STIMOFFSET")
# data119 = read_edf(data119_dir+".asc","STIMONSET","STIMOFFSET")
# data120 = read_edf(data120_dir+".asc","STIMONSET","STIMOFFSET")
# data121 = read_edf(data121_dir+".asc","STIMONSET","STIMOFFSET")
# data122 = read_edf(data122_dir+".asc","STIMONSET","STIMOFFSET")
# data123 = read_edf(data123_dir+".asc","STIMONSET","STIMOFFSET")
# data124 = read_edf(data124_dir+".asc","STIMONSET","STIMOFFSET")
# data125 = read_edf(data125_dir+".asc","STIMONSET","STIMOFFSET")
# data126 = read_edf(data126_dir+".asc","STIMONSET","STIMOFFSET")
# data127 = read_edf(data127_dir+".asc","STIMONSET","STIMOFFSET")
# data128 = read_edf(data128_dir+".asc","STIMONSET","STIMOFFSET")
# data129 = read_edf(data129_dir+".asc","STIMONSET","STIMOFFSET")
# data130 = read_edf(data130_dir+".asc","STIMONSET","STIMOFFSET")



# this is a modified version of the above, which just saves the events
# and does not separate the trials
# we'll use it to determine trials that contain blinks 

data101_plain = read_edf_plain(data101_dir+".asc")
data102_plain = read_edf_plain(data102_dir+".asc")
data103_plain = read_edf_plain(data103_dir+".asc")
data104_plain = read_edf_plain(data104_dir+".asc")
data105_plain = read_edf_plain(data105_dir+".asc")
data106_plain = read_edf_plain(data106_dir+".asc")
data107_plain = read_edf_plain(data107_dir+".asc")
data108_plain = read_edf_plain(data108_dir+".asc")
data109_plain = read_edf_plain(data109_dir+".asc")
data110_plain = read_edf_plain(data110_dir+".asc")
data111_plain = read_edf_plain(data111_dir+".asc")
data112_plain = read_edf_plain(data112_dir+".asc")
data113_plain = read_edf_plain(data113_dir+".asc")
data114_plain = read_edf_plain(data114_dir+".asc")
data115_plain = read_edf_plain(data115_dir+".asc")
data116_plain = read_edf_plain(data116_dir+".asc")
data117_plain = read_edf_plain(data117_dir+".asc")
# data118_plain = read_edf_plain(data118_dir+".asc")
# data119_plain = read_edf_plain(data119_dir+".asc")
# data120_plain = read_edf_plain(data120_dir+".asc")
# data121_plain = read_edf_plain(data121_dir+".asc")
# data122_plain = read_edf_plain(data122_dir+".asc")
# data123_plain = read_edf_plain(data123_dir+".asc")
# data124_plain = read_edf_plain(data124_dir+".asc")
# data125_plain = read_edf_plain(data125_dir+".asc")
# data126_plain = read_edf_plain(data126_dir+".asc")
# data127_plain = read_edf_plain(data127_dir+".asc")
# data128_plain = read_edf_plain(data128_dir+".asc")
# data129_plain = read_edf_plain(data129_dir+".asc")
# data130_plain = read_edf_plain(data130_dir+".asc")


data = [data101, 
        data102, 
        data103, 
        data104, 
        data105,
        data106,
        data107,
        data108,
        data109,
        data110,
        data111,
        data112,
        data113,
        data114,
        data115,
        data116,
        data117,
        # data118,
        # data119,
        # data120,
        # data121,
        # data122,
        # data123,
        # data124,
        # data125,
        # data126,
        # data127,
        # data128,
        # data129,
        # data130
        ]

data_plain = [data101_plain, 
              data102_plain, 
              data103_plain, 
              data104_plain,
              data105_plain,
              data106_plain,
              data107_plain,
              data108_plain,
              data109_plain,
              data110_plain,
              data111_plain,
              data112_plain,
              data113_plain,
              data114_plain,
              data115_plain,
              data116_plain,
              data117_plain,
              # data118_plain,
              # data119_plain,
              # data120_plain,
              # data121_plain,
              # data122_plain,
              # data123_plain,
              # data124_plain,
              # data125_plain,
              # data126_plain,
              # data127_plain,
              # data128_plain,
              # data129_plain,
              # data130_plain
              ]

            
# prefix nrgr = no_regressions
nrgrdur = []
nrgrtime_before = []

# prefix wrgr = accepting regressions
wrgrdur = []
wrgrtime_before = []

nrgrffd = []
nrgrgd = []
nrgrprfix = []

wrgrffd = []
wrgrgd = []
wrgrprfix = []
    
# loop over participants  
for i,dat in enumerate(data):
    nrgrdur_i, nrgrtime_before_i = no_lookback_firstAOI(data[i],data_plain[i])
    wrgrdur_i, wrgrtime_before_i = allowing_lookback_firstAOI(data[i],data_plain[i])
    
    nrgrFFD_i, nrgrGD_i, nrgrfixated_i = ffdgd(nrgrdur_i)
    wrgrFFD_i, wrgrGD_i, wrgrfixated_i = ffdgd(wrgrdur_i)
    
    nrgrdur.append(nrgrdur_i)
    nrgrtime_before.append(nrgrtime_before_i)
    wrgrtime_before.append(wrgrtime_before_i)
    nrgrffd.append(nrgrFFD_i)
    nrgrgd.append(nrgrGD_i)
    nrgrprfix.append(nrgrfixated_i)
    
    wrgrdur.append(wrgrdur_i)
    wrgrffd.append(wrgrFFD_i)
    wrgrgd.append(wrgrGD_i)
    wrgrprfix.append(wrgrfixated_i)

# attach for convenience the single-word and sentence-level statistics
nrgrgd_all = attach_info(nrgrgd, nrgrtime_before)
nrgrffd_all = attach_info(nrgrffd, nrgrtime_before)
nrgrfixations_all = attach_info(nrgrprfix, nrgrtime_before)

wrgrgd_all = attach_info(wrgrgd, wrgrtime_before)
wrgrffd_all = attach_info(wrgrffd, wrgrtime_before)
wrgrfixations_all = attach_info(wrgrprfix, wrgrtime_before)


norm_nrgrgd_all = attach_mean_centred(nrgrgd, nrgrtime_before)
norm_nrgrffd_all = attach_mean_centred(nrgrffd, nrgrtime_before)
norm_nrgrfixations_all = attach_mean_centred(nrgrprfix, nrgrtime_before)

norm_wrgrgd_all = attach_mean_centred(wrgrgd, wrgrtime_before)
norm_wrgrffd_all = attach_mean_centred(wrgrffd, wrgrtime_before)
norm_wrgrfixations_all = attach_mean_centred(wrgrprfix, wrgrtime_before)

##################################################################################################################
##################################################################################################################
##################################################################################################################

# ### HERE ARE PLOTS, uncomment if necessary
# ###### visualization purposes only, right now only one subject (will create code for average)

ax = sns.regplot(data=nrgrffd_all[16][nrgrffd_all[16]['ms']>0], x='Sim',y='ms')
ax.set_title('First Fixation duration - Cloze SemanticSimilarity', fontsize = 15);

bx = sns.regplot(data=nrgrffd_all[16][nrgrffd_all[16]['ms']>0], x='cloze',y='ms')
bx.set_title('First Fixation duration - Cloze', fontsize = 15);

cx = sns.regplot(data=nrgrffd_all[16][nrgrffd_all[16]['ms']>0], x='LogFreq(Zipf)',y='ms')
cx.set_title('First Fixation duration - LogFrequency (Zipf)', fontsize = 15);

dx = sns.regplot(data=norm_nrgrffd_all[16][norm_nrgrffd_all[16]['ms']>0], x='ConcM',y='ms')
dx.set_title('First Fixation duration - Concreteness', fontsize = 15);

ex = sns.regplot(data=norm_nrgrffd_all[16][norm_nrgrffd_all[16]['ms']>0], x='mink3_SM', y='ms')
ex.set_title('First Fixation duration - Sensorimotor', fontsize = 15);

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
# pd.concat(norm_nrgrffd_all).to_csv('C:/Users/fm02/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_ffdpilots_onesemsim1026.csv',index=False)

# pd.concat(norm_nrgrgd_all).to_csv('C:/Users/fm02/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_gdpilots_onesemsim1026.csv',index=False)

# pd.concat(nrgrffd_all).to_csv('C:/Users/User/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/ffdpilots_onesemsim.csv',index=False)

# pd.concat(norm_wrgrffd_all).to_csv('C:/Users/User/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_ffdpilots_withregression_onesemsim.csv',index=False)

# pd.concat(norm_wrgrgd_all).to_csv('C:/Users/User/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_gdpilots_withregression_onesemsim.csv',index=False)

# pd.concat(wrgrffd_all).to_csv('C:/Users/User/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/ffdpilots_withregression_onesemsim.csv',index=False)
