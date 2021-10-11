#!/usr/bin/env python3
# coding: utf8

import os
import pandas as pd
import numpy as np

# Display settings
DISPTYPE = "psychopy"
DISPSIZE = (1280, 1024)

# if dummymode is "True", then pretend the eyetracker is connected; if "False", eyetracker must be there
DUMMYMODE = False

# Background (BGC) and foreground (FGC) colours.
BGC = (128, 128, 128)
FGC = (0,0,0)

# Log file name.
LOGFILENAME = input("Participant number: ")

# assign yes/no buttons:

yesbutton = ["i","I"]
nobutton = ["o","O"]

# Find where THIS file is.
DIR = os.path.dirname(__file__)
DATADIR = os.path.join(DIR, "data", LOGFILENAME, "")

# Find out whether a data directory exists.
if not os.path.isdir(DATADIR):
    os.mkdir(DATADIR)

# Create the path to the log file.
LOGFILE = os.path.join(DATADIR, LOGFILENAME)

KEYLIST = ['space','i', 'I', 'o', 'O']

TRACKERTYPE = 'eyelink' # either 'smi', 'eyelink' or 'dummy' (NB: if DUMMYMODE is True, trackertype will be set to dummy automatically)
#TRACKERTYPE = 'dummy'

STIMDUR = 5000

ALLOW_ESC_KILL = True

# participants will press i=Yes or o=No in comprehension questions
klist = ["i","I","o","O"]

# stimuli = pd.read_csv('SHORTSentANDQuest.txt', sep='\t', header=0)
stimuli = pd.read_csv('SentANDQuest.txt', sep='\t', header=0)
# stimuli = pd.read_csv('short.txt', sep='\t', header=0)

examples = pd.DataFrame(['This sentence is an example, when you finish reading look at the right bottom corner.',
						 'This is another example, when you finish reading look at the right bottom corner.',
						 'The River Cam flows into the Great Ouse to the south of Ely.',
						 'Geoffrey Chaucer is widely considered the greatest English poet of the Middle Ages.'], columns=['Sent'])