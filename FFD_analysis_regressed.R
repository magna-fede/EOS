# this script analyzes only trials with regressions
# check if those trials are different

library(readxl)
library(lme4)
library(lmerTest)
library(languageR)
library(lattice)
library(BayesFactor)
library(dplyr)

# import dataset
FFD2 <- read.csv('C:/Users/fm02/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_ffd_41.csv')
FFD2 <- FFD2 %>% dplyr::filter(regressed==1)
# consider only values greater than 0
# previously selected only fixations 80-600ms long
FFD2 <- FFD2[FFD2$ms != 0, ]

colnames(FFD2)

# check which primary factors affect fixation durations

# let's try some of the predictos influence on fixation durations

############ trial number (in the sense of order) #######

trialsFFD2 = lmer(ms ~ trialnr + (1|ID) + (1|Subject), data = FFD2)
summary(trialsFFD2)
# The order of the trials does not seem to affect FFD, so not including it in the model

############ log frequency Zipf value #######

freqFFD2 = lmer(ms ~ LogFreqZipf + (1|ID) + (1|Subject), data = FFD2)
summary(freqFFD2)
# LogFreqZipf affects FFD , so including it

############ length of the word (in the sense of number of characters) #######
LENFFD2 = lmer(ms ~ LEN + (1|ID) + (1|Subject), data = FFD2)
summary(LENFFD2)
# Length does not seem to affect FFD, so not including it in the model

############ preceding word logFreq distance #######
precFFD2 = lmer(ms ~ PRECEDING_LogFreqZipf + (1|ID) + (1|Subject), data = FFD2)
summary(precFFD2)
# preceding word logFreq does not seem to affect FFD, so not including it

############ position in the sentence #######
posFFD2 = lmer(ms ~  Position + (1|ID) + (1|Subject), data = FFD2)
summary(posFFD2)
# Position in the sentence seems to affect FFD, so including it

############ time before FFD #######
tbFFD2 = lmer(ms ~  time_before_ff + (1|ID) + (1|Subject), data = FFD2)
summary(tbFFD2)
# Time before FFD does not seem to affect FFD, so not including it in the model

######## this is our basic model ###########
lmeBasic = lmer(ms ~ trialnr + PRECEDING_LogFreqZipf + time_before_ff + (1|ID) + (1|Subject), data = FFD2)
summary(lmeBasic)

################ introduce Predictability(a priori), cloze, and semanticpredictability(=sim) ###############
lmeOnlyPred = lmer(ms ~ trialnr + PRECEDING_LogFreqZipf + time_before_ff + Predictability +
                     (1|ID) + (1|Subject), data = FFD2)
summary(lmeOnlyPred)

lmeOnlyCloze = lmer(ms ~ trialnr + PRECEDING_LogFreqZipf + time_before_ff + cloze +
                      (1|ID) + (1|Subject), data = FFD2)
summary(lmeOnlyCloze)

lmeOnlySemSim = lmer(ms ~ trialnr + PRECEDING_LogFreqZipf + time_before_ff +  Sim +
                       (1|ID) + (1|Subject), data = FFD2)
summary(lmeOnlySemSim)

anova(lmeBasic,lmeOnlyPred)
anova(lmeBasic, lmeOnlyCloze)
anova(lmeBasic, lmeOnlySemSim)

# All are significant, but cloze probability explains the most variance.

################ check other confounds ###############

# both similarity and plausibility seem to affect FFD

lmesim = lmer(ms ~ trialnr + PRECEDING_LogFreqZipf + time_before_ff + similarity + (1|ID) + (1|Subject), data = FFD2)
summary(lmesim)
lmeplau = lmer(ms ~ trialnr + PRECEDING_LogFreqZipf + time_before_ff + plausibility + (1|ID) + (1|Subject), data = FFD2)
summary(lmeplau)
anova(lmeBasic,lmeplau)

# but in both cases, it seems that when including cloze as a predictor, it incorporates their variance


################ introduce Concreteness ###############
onlyConc = lmer(ms ~ trialnr + PRECEDING_LogFreqZipf + time_before_ff + ConcM + (1|ID) + (1|Subject), data = FFD2)
summary(onlyConc)

anova(lmeBasic,onlyConc)

# Concreteness affects fixation durations, worth including it in the model

################# let's check sensorimotor strength
onlySM = lmer(ms ~ trialnr + PRECEDING_LogFreqZipf + time_before_ff + mink3_SM + (1|ID) + (1|Subject), data = FFD2)
summary(onlySM)

# also sensorimotor strength affects fixation durations, but less variance explained

sjPlot::tab_model(lmeplau)
sjPlot::plot_model(lmeplau)
