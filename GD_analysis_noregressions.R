# this script analyses which factors affect GD in trials without regressions

# import stuff
library(readxl)
library(lme4)
library(lmerTest)
library(languageR)
library(lattice)
library(BayesFactor)

# import dataset
GD2 <- read.csv('C:/Users/fm02/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_nrgr_gd_41.csv')
GD2 <- GD2[GD2$ms != 0, ]

# GD2 <- GD2[!(GD2$Subject== 3 | GD2$Subject== 14 | GD2$Subject== 33 | GD2$Subject== 34 ),]
# let's try some of the predictors influence on fixation duration

############ trial number (in the sense of order) #######

trialsGD2 = lmer(ms ~ trialnr + (1|ID) + (1|Subject), data = GD2)
summary(trialsGD2)
# The order of the trials does not affect GD, so not including it

############ log frequency Zipf value #######

freq = lmer(ms ~ LogFreqZipf + (1|ID) + (1|Subject), data = GD2)
summary(freq)
# LogFreqZipf affects GD , so including it

############ length of the word (in the sense of number of characters) #######
LENGD2 = lmer(ms ~ LEN + (1|ID) + (1|Subject), data = GD2)
summary(LENGD2)
# length affects GD, so including it 

############ preceding word logFreq distance #######
precGD2 = lmer(ms ~ PRECEDING_LogFreqZipf + (1|ID) + (1|Subject), data = GD2)
summary(precGD2)
# preceding word logFreq seems to affect GD, so including it

############ position in the sentence #######
posGD2 = lmer(ms ~ Position + (1|ID) + (1|Subject), data = GD2)
summary(posGD2)
# Position affects marginally GD, so try both with and without not including it

######## create our basic model ###########
lmeBasicGD2 = lmer(ms ~ LogFreqZipf + LEN + PRECEDING_LogFreqZipf + Position + (1|ID) + (1|Subject), data = GD2)
summary(lmeBasicGD2)
# in this model it seems that PRECEDING_LogFreqZipf variance if better explained
# by other variables. so excluding that but keeping Position in

######## this is our basic model ###########
lmeBasicGD2 = lmer(ms ~ LogFreqZipf + LEN +  Position + (1|ID) + (1|Subject), data = GD2)
summary(lmeBasicGD2)

################ introduce Predictability(a priori), cloze, and semanticpredictability(=sim) ###############
lmeOnlyPred = lmer(ms ~ LogFreqZipf + LEN + Position +  Predictability +
                     (1|ID) + (1|Subject), data = GD2)
summary(lmeOnlyPred)

lmeOnlyCloze = lmer(ms ~ LogFreqZipf + LEN +  Position + cloze +
                      (1|ID) + (1|Subject), data = GD2)
summary(lmeOnlyCloze)

lmeOnlySemSim = lmer(ms ~ LogFreqZipf + LEN +  Position + Sim +
                       (1|ID) + (1|Subject), data = GD2)
summary(lmeOnlySemSim)

anova(lmeBasicGD2,lmeOnlyPred)
anova(lmeBasicGD2, lmeOnlyCloze)
anova(lmeBasicGD2, lmeOnlySemSim)

# In this case, all are significant, but cloze semantic similarity explains the most variance
# (not cloze only, even though same order of magnitude).

################ introduce Concreteness ###############
onlyConc = lmer(ms ~ LogFreqZipf + LEN + Position +  ConcM + (1|ID) + (1|Subject), data = GD2)
summary(onlyConc)
# concreteness does not affect GD
anova(lmeBasicGD2,onlyConc)
# no effect of concreteness (only marginal)

################# let's check sensorimotor strength
onlySM = lmer(ms ~ LogFreqZipf + LEN + Position + mink3_SM + (1|ID) + (1|Subject), data = GD2)
summary(onlySM)
# also Sensorimotor strength does not have an effect on GD

additive_ConcM.Sim = lmer(ms ~ LogFreqZipf + LEN + Sim + ConcM + (1|ID) + (1|Subject), data = GD2)

summary(interaction_ConcM.Sim)
summary(additive_ConcM.Sim) 
anova(interaction_ConcM.Sim,additive_ConcM.Sim )

# concreteness does not affect Gaze duration