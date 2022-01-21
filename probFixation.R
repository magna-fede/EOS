# this script takes as input whether a word fixated or not
# We want to inspect which factors make a word more likely to be fixated.

# import stuff
library(readxl)
library(lme4)
library(lmerTest)
library(languageR)
library(lattice)
library(BayesFactor)

# import the dataset
FFD3 <- read.csv('C:/Users/fm02/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_nrgr_ffd_41.csv')

# 1 if word was fixated, 0 if not
FFD3$ms01 <- ifelse(FFD3$ms > 0, 1, 0)
# FFD3 <- FFD3[!(FFD3$Subject== 3 | FFD3$Subject== 14 | FFD3$Subject== 33 | FFD3$Subject== 34 ),]

############ log frequency Zipf value #######

freq = lmer(ms01 ~ LogFreqZipf + (1|ID) + (1|Subject), data = FFD3)
summary(freq)
# LogFreqZipf affects p(fixation), so including it

############ length of the word (in the sense of number of characters) #######
LENFFD3 = lmer(ms01 ~ LEN + (1|ID) + (1|Subject), data = FFD3)
summary(LENFFD3)
# length affects p(fixation), so including it 

############ preceding word logFreq distance #######
precFFD3 = lmer(ms01 ~ PRECEDING_LogFreqZipf + (1|ID) + (1|Subject), data = FFD3)
summary(precFFD3)
# preceding word logFreq does not seem to affect GD, so not including it

############ time before first fixation #######
tbfFFD3 = lmer(ms01 ~ time_before_ff + (1|ID) + (1|Subject), data = FFD3)
summary(tbfFFD3)
# time before FF affects p(fix), so including it

############ Position in the Sentence #######
posFFD3 = lmer(ms01 ~  Position + (1|ID) + (1|Subject), data = FFD3)
summary(posFFD3)
# Position affects p(fix), so including it

######## this is our basic model ###########
lmeBasicFFD3 = lmer(ms01 ~ LogFreqZipf + LEN + Position + time_before_ff + (1|ID) + (1|Subject), data = FFD3)
summary(lmeBasicFFD3)

# not excluding frequency because even if no longer significant here, it is when including predictability variables
######## this is our basic model ###########
lmeBasicFFD3 = lmer(ms01 ~ LogFreqZipf + LEN + Position + time_before_ff + (1|ID) + (1|Subject), data = FFD3)
summary(lmeBasicFFD3)


################ introduce Predictability(a priori), cloze, and semanticpredictability(=sim) ###############

lmeOnlyPred = lmer(ms01 ~ LogFreqZipf + LEN + Position + time_before_ff + Predictability +
                     (1|ID) + (1|Subject), data = FFD3)
summary(lmeOnlyPred)

lmeOnlyCloze = lmer(ms01 ~ LogFreqZipf + LEN + Position + time_before_ff + cloze +
                      (1|ID) + (1|Subject), data = FFD3)
summary(lmeOnlyCloze)

lmeOnlySemSim = lmer(ms01 ~ LogFreqZipf + LEN + Position + time_before_ff + Sim +
                       (1|ID) + (1|Subject), data = FFD3)
summary(lmeOnlySemSim)

anova(lmeBasicFFD3,lmeOnlyPred)
anova(lmeBasicFFD3, lmeOnlyCloze)
anova(lmeBasicFFD3, lmeOnlySemSim)

# In this case, cloze semantic similarity affect probability of fixating a word first pass.

################ introduce Concreteness ###############
onlyConc = lmer(ms01 ~ LogFreqZipf + LEN + Position + time_before_ff + ConcM + (1|ID) + (1|Subject), data = FFD3)
summary(onlyConc)

anova(lmeBasicFFD3,onlyConc)
# concreteness seem to affect the probability to fixate a word (more abstract words more likely to be fixated)

################# let's check sensorimotor strength
onlySM = lmer(ms01 ~ LogFreqZipf + LEN + Position + time_before_ff + mink3_SM + (1|ID) + (1|Subject), data = FFD3)
summary(onlySM)
# sensorimotor strength no

################ just for fun, let's look at the interaction
interaction_ConcM.Sim = lmer(ms01 ~ LogFreqZipf + LEN + Position + time_before_ff + Sim*ConcM + (1|ID) + (1|Subject), data = FFD3)
additive_ConcM.Sim = lmer(ms01 ~ LogFreqZipf + LEN + Position + time_before_ff + Sim + ConcM + (1|ID) + (1|Subject), data = FFD3)

summary(additive_ConcM.Sim)

