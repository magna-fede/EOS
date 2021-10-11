#!/usr/bin/env python3
# coding: utf8

import random

from constants import *

from pygaze.display import Display
from pygaze.screen import Screen
from pygaze.sound import Sound
from pygaze.keyboard import Keyboard
from pygaze.mouse import Mouse
from pygaze.eyetracker import EyeTracker
from pygaze.logfile import Logfile
import pygaze.libtime as timer
import json
from psychopy import visual

# Create a new Display instance.
disp = Display()

# Create eyetracker object
tracker = EyeTracker(disp)

# create keyboard object
keyboard = Keyboard()

# Create a new Logfile instance.
log = Logfile()
log.write(["trialnr", "IDstim", "target", "stimulus", "size_stim", "fix_onest", \
    "stim_onset", "stim_offset", "Question", "Answer"])

# Create an internal dict for all the trials where the gaze position (gpos) for each timepoint(t) of each trial (i)
# will be exported {trial: {t: (x,y)}}

logpos = {}

# define the position of the fixation cross for starting each trial...
startpos = (50,DISPSIZE[1]/2)
# ... and the *centre* position of the square for the end of each trial...
endpos = (DISPSIZE[0]-15,DISPSIZE[1]-15)
# ... and *top-left corner* (where to start drawing the square) of the right-bottom corner
endpos_sq = (DISPSIZE[0]-30,DISPSIZE[1]-30)

# Initialise a blank screen.
blankscr = Screen()

###########################################
######  INSTRUCTIONS ######################
###########################################

# Create a new Screen instance.
scr = Screen()
scr.draw_text("Welcome!", fontsize=100, \
    colour=(255,100,100))

# Pass the screen to the display.
disp.fill(scr)
disp.show()
timer.pause(1000)

# Create a new Screen instance.
instscr0 = Screen()
instscr0.draw_text("""
In this experiment you will see some sentences on the screen. Your task is to carefully read them.\n
Before the start of each sentence you will see a fixation cross on the left.
You need to look at that until it is replaced by a sentence.\n\n
Sometimes it can happen that the sentence doesn't appear, this is because the eyetracker does not capture the position of your eyes correclty.
If this happens, press Q, to calibrate the eyetracker.\n\n
When you finish reading the sentence, look at the black square at the right bottom corner of the screen.\n\n
Let's do a calibration of the eyetracker and do some examples. (press space to start)""", \
    font='consolas', fontsize=20,)

# this gets the psychopy screen, to change the wrapWidth=at which point go to the next line
# indeed, pygaze does not allow for this manipulation directly, but we need to use psychopy
text0 = instscr0.screen[-1]
text0.wrapWidth = DISPSIZE[0]-100
# and present back the modified screen
instscr0.screen[-1] = text0

# present it on screen
disp.fill(instscr0)
disp.show()

# flush the keyboard
keyboard.get_key(keylist=None, timeout=None, flush=True)

# Open the calibration menu.
tracker.calibrate()
# Upon returning from the calibration, restart recording.
timer.pause(5)

# start eye tracking
tracker.start_recording()

####### EXAMPLES SENTENCES ###############################################################
##########################################################################################

# get examples from constants
for i, trial in enumerate(examples['Sent']):

    # # drift correction
    # checked = False
    # while not checked:
    #     disp.fill(fixscr)
    #     disp.show()
    #     checked = tracker.drift_correction()

    # start eye tracking
    tracker.start_recording()
    
    # Log the trial start to the tracker (wait 5 milliseconds first, to allow
    # the trial to start).
    timer.pause(5)
    
    # Draw the stimulus for this trial.
    stimscr = Screen()
    stimscr.draw_rect(colour="black", x=endpos_sq[0], y=endpos_sq[1], fill=True, w=30, h=30)

    stimscr.draw_text(trial, fontsize=20)
 
    # Avoid newline unless sentence is same length of display + set the font (monospaced)
    textstim = stimscr.screen[-1]
    textstim.wrapWidth = DISPSIZE[0]
    textstim.font = 'consolas'
    stimscr.screen[-1] = textstim
    
    disp.fill(stimscr)

    sizestim = stimscr.screen[-1].boundingBox 
    
    # Draw the fixation screen.
    fixscr = Screen()
    fixscr.draw_fixation(fixtype="cross", diameter=20, pw=3, pos=startpos)
    # Show the fixation screen on the monitor.
    disp.fill(fixscr)
    fix_onset = disp.show()
    tracker.log("FIXONSET")
    # Wait for a wee bit.
    timer.pause(random.randint(250, 750))

    # Flush the keyboard.
    key, presstime = keyboard.get_key(keylist=None, timeout=1, flush=True)
    
    # Make sure the participant is looking at the fixation cross
    centre_fix = False

    t_prev = timer.get_time()
    t = timer.get_time()

    # check that the participant is looking at the fixation cross in startpos for at least 200ms
    while not centre_fix:
        # Get the current gaze position.
        pos = tracker.sample()
        # Check if fixating the cross.
        if (((pos[0]-startpos[0])**2 + (pos[1]-startpos[1])**2) < (30**2)):
            # get starting time
            t = timer.get_time()
            if t - t_prev>=200:
                centre_fix = True
        else:
            t_prev = timer.get_time()

        # Check if a key was pressed.
        key, presstime = keyboard.get_key(keylist=None, timeout=1, \
            flush=False)
        # Force continue when pressing "B".
        # use this key as it is unlikely pressed accidentally (as space might be)
        if key in ["b", "B"]:
            centre_fix = True
        # Start a calibration on Q.
        elif key in ["q", "Q"]:
            # Stop recording to allow the calibration to start.
            tracker.stop_recording()
            timer.pause(5)
            # Open the calibration menu.
            tracker.calibrate()
            # Upon returning from the calibration, restart recording.
            timer.pause(5)
            tracker.start_recording()
            # Show the fixation screen again.
            disp.fill(fixscr)
            fixonset = disp.show()
            timer.pause(random.randint(750, 1250))


    # Show the stimulus .
    disp.fill(stimscr)
    stim_onset = disp.show()

    # get the size of the stimulus on screen (centred in the middle of the screen)
    # measures of each character is (11,23)
    # sizestim = stimscr.screen[-1].boundingBox
    # get the position of the target word INSIDE the boundingBox

    # Initialise the dict for the trial {t: (x,y)} so that we record gaze position for each timepoint
    logpos_trial = {} 

    # Record eye position until space bar is pressed
    t_in = 0
    fix_end = False
    while not fix_end:
        # Get the current gaze position.
        pos = tracker.sample()
        # Get the current timestamp.
        t = timer.get_time()
        # Save {t: (x,y)}
        logpos_trial[t] = pos
        # Check if a key is pressed
        key, presstime = keyboard.get_key(keylist=None, timeout=1, flush=False)
        # Get out when participants look at the square on the bottom right corner
        if (((pos[0]-endpos[0])**2 + (pos[1]-endpos[1])**2) < (90**2)):
            # count how much time they are looking at the square (100ms are enough)
            t_in += 1
            if t_in >=100:
                fix_end = True
        else:
            # or set to zero if not looking there
            t_in = 0
        # finish trial if they press space            
        if key == 'space':
            break
        # or close the experiment if they press Esc    
        elif (key in ['Escape', 'escape', 'Esc', 'esc']) and ALLOW_ESC_KILL:
            log.close()
            tracker.close()
            disp.close()
            raise Exception("ESCAPE: Escape key pressed.")

    # Remove the stimulus screen.
    disp.fill(blankscr)
    stim_offset = disp.show()

    # Flush the keyboard.
    key, presstime = keyboard.get_key(keylist=None, timeout=1, flush=True)
            
    # Inter-trial interval.
    timer.pause(1000)

disp.fill(blankscr)
disp.show()
timer.pause(2000)

instscr2 = Screen()
instscr2.draw_text("""
It is important that you read carefully each sentence as, after some trials, a question about the previous sentence will appear.\n
You will need to respond using the buttons 'i'='YES' and 'o'='NO'.\n\n
Please, keep your fingers positioned on the keys I, O, and Q. \n
You should avoid moving your head and always look at the screen.\n
Here is an example.  (press space to start)""",  font='consolas', fontsize=20)

text2 = instscr2.screen[-1]
text2.wrapWidth = DISPSIZE[0]-100
instscr2.screen[-1] = text2

disp.fill(instscr2)
disp.show()

key = keyboard.get_key(keylist=None, timeout=None, flush=True)

checkINST=True

while checkINST==True:
    quest0scr = Screen()
    quest0scr.draw_text("This is an example. Press the Yes button to continue.", fontsize=20)
    text0stim = quest0scr.screen[-1]
    text0stim.wrapWidth = DISPSIZE[0]
    quest0scr.screen[-1] = text0stim
    quest0scr.draw_text(yesbutton[0], font='consolas', fontsize=20, pos=(100,DISPSIZE[1]-177))
    quest0scr.draw_text("Yes", font='consolas', fontsize=20, pos=(100,DISPSIZE[1]-200))
    quest0scr.draw_text(nobutton[0], font='consolas', fontsize=20, pos=(DISPSIZE[0]-200,DISPSIZE[1]-177))
    quest0scr.draw_text("No", font='consolas', fontsize=20, pos=(DISPSIZE[0]-200,DISPSIZE[1]-200))
    disp.fill(quest0scr)
    disp.show()

    key, presstime = keyboard.get_key(keylist=klist, timeout=None, flush=True)

    if key in yesbutton:
        instscr3 = Screen()
        instscr3.draw_text("Well done, we are ready to start the experiment! Press any key to start",  font='consolas', fontsize=20,)
        disp.fill(instscr3)
        disp.show()
        checkINST = False
    else:
        instscr3 = Screen()
        instscr3.draw_text("You pressed the wrong button. Let's try again!",  font='consolas', fontsize=20)
        disp.fill(instscr3)
        disp.show()
        timer.pause(2000)
        checkINST = True

key = keyboard.get_key(keylist=None, timeout=None, flush=True)

# Create a list of all trials.
trials = []
# Add all the words.
for i in range(len(stimuli)):
    t = {}
    t["stimulus"] = stimuli.Sent[i]
    t["IDstim"] = stimuli.ID[i]
    t["target"] = stimuli.Word[i]
    t["Question"] = stimuli.Question[i]
    t["Answer"] = str(stimuli.Ans[i])
    trials.append(t)

# Randomise the order.
random.shuffle(trials)

###########################################
######  EXPERIMENT ########################
###########################################

# Loop through all trials.
for i, trial in enumerate(trials):
   
    if i % 40 == 0:
        tracker.stop_recording()
        timer.pause(5)
        # Open the calibration menu.
        tracker.calibrate()
        # Upon returning from the calibration, restart recording.
        timer.pause(5)

    # start eye tracking
    tracker.start_recording()
    
    # Log the trial start to the tracker (wait 5 milliseconds first, to allow
    # the trial to start).
    timer.pause(5)
    
    tracker.log("START TRIAL %d" % i)

    # Draw the stimulus for this trial.
    stimscr = Screen()
    stimscr.draw_rect(colour="black", x=endpos_sq[0], y=endpos_sq[1], fill=True, w=30, h=30)

    stimscr.draw_text(trial["stimulus"], fontsize=20)
 
    # Avoid newline unless sentence is same length of display + set the font (monospaced)
    textstim = stimscr.screen[-1]
    textstim.wrapWidth = DISPSIZE[0]
    textstim.font = 'consolas'
    stimscr.screen[-1] = textstim
    
    disp.fill(stimscr)

    # get the size of the stimulus on screen (centred in the middle of the screen)
    # measures of each character is (11,23)

    sizestim = stimscr.screen[-1].boundingBox 
    
    # Draw the fixation screen.
    fixscr = Screen()
    fixscr.draw_fixation(fixtype="cross", diameter=20, pw=3, pos=startpos)
    # Show the fixation screen on the monitor.
    disp.fill(fixscr)
    fix_onset = disp.show()
    tracker.log("FIXONSET")
    # Wait for a wee bit.
    timer.pause(random.randint(250, 750))

    # Flush the keyboard.
    key, presstime = keyboard.get_key(keylist=None, timeout=1, flush=True)
    
    # Make sure the participant is looking at the fixation cross
    centre_fix = False

    t_prev = timer.get_time()
    t = timer.get_time()
    while not centre_fix:
        # Get the current gaze position.
        pos = tracker.sample()
        # Check if fixating the cross.
        if (((pos[0]-startpos[0])**2 + (pos[1]-startpos[1])**2) < (30**2)):
            # get starting time
            t = timer.get_time()
            if t - t_prev>=200:
                centre_fix = True
        else:
            t_prev = timer.get_time()

        # Check if a key was pressed.
        key, presstime = keyboard.get_key(keylist=None, timeout=1, \
            flush=False)
        # Force continue on a space press.
        if key in ["b", "B"]:
            centre_fix = True
        # Start a calibration on Q.
        elif key in ["q", "Q"]:
            # Stop recording to allow the calibration to start.
            tracker.stop_recording()
            timer.pause(5)
            # Open the calibration menu.
            tracker.calibrate()
            # Upon returning from the calibration, restart recording.
            timer.pause(5)
            tracker.start_recording()
            # Show the fixation screen again.
            disp.fill(fixscr)
            fixonset = disp.show()
            tracker.log("FIXONSET")
            timer.pause(random.randint(750, 1250))


    # Show the stimulus .
    disp.fill(stimscr)
    stim_onset = disp.show()

    # get the position of the target word INSIDE the boundingBox
    targetinsize = (trial['stimulus'].index(trial['target']), (trial['stimulus'].index(trial['target'])+len(trial['target'])))

    # save all the important info for the analysis on tracker.log (save as "MSG" in the EDF output file)
    tracker.log("STIMONSET")
    tracker.log(f"ID STIMULUS: {trial['IDstim']}")
    tracker.log(f"STIMULUS: {trial['stimulus']}")
    tracker.log(f"SIZE OF THE STIMULUS: {sizestim}")
    tracker.log(f"NUMBER OF CHARACTERS: {len(trial['stimulus'])}")
    tracker.log(f"POS TARGET INSIDE BOX: {targetinsize}")

    t = timer.get_time()

    # Initialise the dict for the trial {t: (x,y)} so that we record gaze position for each timepoint
    logpos_trial = {} 

    # Record eye position until space bar is pressed
    t_in = 0
    fix_end = False
    while not fix_end:
        # Get the current gaze position.
        pos = tracker.sample()
        # Get the current timestamp.
        t = timer.get_time()
        # Save {t: (x,y)}
        logpos_trial[t] = pos
        # Check if a key is pressed
        key, presstime = keyboard.get_key(keylist=None, timeout=1, flush=False)
        # Get out when looking at the right bottom corner square
        if (((pos[0]-endpos[0])**2 + (pos[1]-endpos[1])**2) < (90**2)):
            # get starting time
            t_in += 1
            if t_in >=100:
                fix_end = True
        else:
            t_in = 0
        # or if "space" is pressed            
        if key == 'space':
            break
        elif (key in ['Escape', 'escape', 'Esc', 'esc']) and ALLOW_ESC_KILL:
            log.close()
            tracker.close()
            disp.close()
            raise Exception("ESCAPE: Escape key pressed.")

    # Remove the stimulus screen.
    disp.fill(blankscr)
    stim_offset = disp.show()
    tracker.log("STIMOFFSET")

    # Append trial recording
    logpos[i] = logpos_trial

    if str(trial["Question"]) != "0": # compare zero as a string (not integer) on both ends
        questscr = Screen()
        questscr.draw_text(trial["Question"], fontsize=20)
        questscr.draw_text(yesbutton[0], font='consolas', fontsize=20, pos=(100,DISPSIZE[1]-377))
        questscr.draw_text("Yes", font='consolas', fontsize=20, pos=(100,DISPSIZE[1]-400))
        questscr.draw_text(nobutton[0], font='consolas', fontsize=20, pos=(DISPSIZE[0]-200,DISPSIZE[1]-377))
        questscr.draw_text("No", font='consolas', fontsize=20, pos=(DISPSIZE[0]-200,DISPSIZE[1]-400))
        disp.fill(questscr)
        disp.show()

        while key not in klist:
            key, presstime = keyboard.get_key(keylist=["i","I","o","O"], timeout=1, flush=False)
        if ((key in yesbutton) & (trial["Answer"]=="1")) or ((key in nobutton) & (trial["Answer"]=="0")):
            trial["Answer"]="Correct"
        elif ((key in yesbutton) & (trial["Answer"]=="0")) or ((key in nobutton) & (trial["Answer"]=="1")):
            trial["Answer"]="Incorrect"

    # Record the response.
    log.write([i, trial["IDstim"],trial["target"],trial["stimulus"], sizestim, fix_onset,\
        stim_onset, stim_offset, trial["Question"], trial["Answer"]])

    # Flush the keyboard.
    key, presstime = keyboard.get_key(keylist=None, timeout=1, flush=True)
            
    # Inter-trial interval.
    disp.fill()
    disp.show()
    timer.pause(1000)

# # # # #
# CLOSE
textscr = Screen()
txt = "The experiment is finished. Thanks for participating!"
textscr.draw_text(text=txt, fontsize=24)
disp.fill(textscr)
disp.show()
timer.pause(2000)

# Show a screen that notifies the participant that the task is closing down.
textscr.clear()
textscr.draw_text(text="Saving data; don't touch anything, please!", \
    fontsize=24)
disp.fill(textscr)
disp.show()

# Flush the keyboard (to get rid of any queued key presses).
keyboard.get_key(keylist=None, timeout=1, flush=True)

# Save gaze position
with open(DATADIR+LOGFILENAME+'_outputfile.json', 'w') as fout:
    json.dump(logpos, fout)

# Neatly close the logfile.
log.close()
# and the eyetracker
tracker.close()

# Exit the display.
disp.close()

timer.expend()