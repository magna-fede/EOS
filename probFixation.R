# this script takes as input whether a word fixated or not
# We want to inspect which factors make a word more likely to be fixated.

# CAREFUL! the input data does not distinguish between words that have been skipped altogether
# and words that have been skipped on first pass, but where fixated later
# (e.g., in python preprocessing script words that have never been fixated are reported as an empty series,
# whereas words that were not fixated on first pass are reported as string).
# Therefore, this categorization refers to words that were skipped on first pass reading.
# if you want to make a distinction between skipped vs. skipped only in first pass need to change
# the py analysis script.

# import stuff
library(tidyverse)
library(readxl)
library(lme4)
library(lmerTest)
library(languageR)
library(lattice)
library(BayesFactor)
library(ggeffects)
library(ggplot2)

# import the dataset
GD3 <- read.csv('C:/Users/fm02/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_gd_41_220225.csv')
# using different dataset because not considering as skipped words that were fixated for less than 80 or more than 600
# (because in preprocessing script are turned to zero)
# so if a word's fixation is anomalous, it's still considered as fixated.

# 1 if word was fixated, 0 if not
GD3$ms01 <- ifelse(GD3$ms > 0, 1, 0)
# GD3 <- GD3[!(GD3$Subject== 3 | GD3$Subject== 14 | GD3$Subject== 33 | GD3$Subject== 34 ),]

############ log frequency Zipf value #######

freq = lmer(ms01 ~ LogFreqZipf + (1|ID) + (1|Subject), data = GD3)
summary(freq)
# LogFreqZipf affects p(fixation), so including it

############ length of the word (in the sense of number of characters) #######
LENGD3 = lmer(ms01 ~ LEN + (1|ID) + (1|Subject), data = GD3)
summary(LENGD3)
# length affects p(fixation), so including it 

############ preceding word logFreq distance #######
precGD3 = lmer(ms01 ~ PRECEDING_LogFreqZipf + (1|ID) + (1|Subject), data = GD3)
summary(precGD3)
# preceding word logFreq does not seem to affect GD, so not including it

############ Position in the Sentence #######
posGD3 = lmer(ms01 ~  Position + (1|ID) + (1|Subject), data = GD3)
summary(posGD3)
# Position doesn't affect p(fix), so not including it

######## this is our basic model ###########
lmeBasicGD3 = lmer(ms01 ~ LogFreqZipf + LEN + Position + (1|ID) + (1|Subject), data = GD3)
summary(lmeBasicGD3)

################ introduce Predictability(a priori), cloze, and semanticpredictability(=sim) ###############

lmeOnlyPred = lmer(ms01 ~ LogFreqZipf + LEN + + Predictability +
                     (1|ID) + (1|Subject), data = GD3)
summary(lmeOnlyPred)

lmeOnlyCloze = lmer(ms01 ~ LogFreqZipf + LEN + cloze +
                      (1|ID) + (1|Subject), data = GD3)
summary(lmeOnlyCloze)

lmeOnlySemSim = lmer(ms01 ~ LogFreqZipf + LEN + Sim +
                       (1|ID) + (1|Subject), data = GD3)
summary(lmeOnlySemSim)

anova(lmeBasicGD3,lmeOnlyPred)
anova(lmeBasicGD3, lmeOnlyCloze)
anova(lmeBasicGD3, lmeOnlySemSim)

# In this case, cloze semantic similarity affect probability of fixating a word first pass.

################ introduce Concreteness ###############
onlyConc = lmer(ms01 ~ LogFreqZipf + LEN + ConcM + (1|ID) + (1|Subject), data = GD3)
summary(onlyConc)

anova(lmeBasicGD3,onlyConc)
# concreteness seems not to affect the probability to fixate a word

################# let's check sensorimotor strength
onlySM = lmer(ms01 ~ LogFreqZipf + LEN + Position + mink3_SM + (1|ID) + (1|Subject), data = GD3)
anova(lmeBasicGD3,onlySM)
# sensorimotor strength no

################ just for fun, let's look at the interaction
additive_ConcM.Sim = lmer(ms01 ~ LogFreqZipf + LEN + Position + Sim + ConcM + (1|ID) + (1|Subject), data = GD3)
interactive_ConcM.Sim = lmer(ms01 ~ LogFreqZipf + LEN + Position + Sim * ConcM + (1|ID) + (1|Subject), data = GD3)

summary(additive_ConcM.Sim)
summary(interactive_ConcM.Sim)

sjPlot::tab_model(additive_ConcM.Sim)
sjPlot::plot_model(additive_ConcM.Sim)

sjPlot::tab_model(interactive_ConcM.Sim,pred.labels = c('Intercept',
                                                       'Frequency (Zipf)',
                                                       'Length',
                                                       'Position',
                                                       'Predictability',
                                                       'Concreteness',
                                                       'Predictability*Concreteness'))

############################### exploratory #############################
expl = lmer(ms01 ~ LogFreqZipf + A_MeanSum + LEN + Sim + ConcM + Position + (1|ID) + (1|Subject), data = GD3)
summary(expl)

expl2 = lmer(ms01 ~ LogFreqZipf + V_MeanSum + LEN + Sim + Position + (1|ID) + (1|Subject), data = GD3)
summary(expl2)

# careful! more likely to fixate more arousing words
anova(additive_ConcM.Sim,expl)
