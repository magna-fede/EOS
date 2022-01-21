# this script analyses which factors affect FFD in trials without regressions

# import stuff
library(readxl)
library(lme4)
library(lmerTest)
library(languageR)
library(lattice)
library(BayesFactor)

# import dataset
FFD2 <- read.csv('C:/Users/fm02/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_nrgr_ffd_41.csv')
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
lmeBasic = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + (1|ID) + (1|Subject), data = FFD2)
summary(lmeBasic)

################ introduce Predictability(a priori), cloze, and semanticpredictability(=sim) ###############
lmeOnlyPred = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + Predictability +
                     (1|ID) + (1|Subject), data = FFD2)
summary(lmeOnlyPred)

lmeOnlyCloze = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + cloze +
                      (1|ID) + (1|Subject), data = FFD2)
summary(lmeOnlyCloze)

lmeOnlySemSim = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position +  Sim +
                       (1|ID) + (1|Subject), data = FFD2)
summary(lmeOnlySemSim)

anova(lmeBasic,lmeOnlyPred)
anova(lmeBasic, lmeOnlyCloze)
anova(lmeBasic, lmeOnlySemSim)

# All are significant, but cloze probability explains the most variance.

################ check other confounds ###############
lmesim = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + similarity + (1|ID) + (1|Subject), data = FFD2)
summary(lmesim)

lmeplau = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + plausibility + (1|ID) + (1|Subject), data = FFD2)
summary(lmeplau)

# both similarity and plausibility seem to affect FFD

lmesimcloze = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + cloze + similarity + (1|ID) + (1|Subject), data = FFD2)
summary(lmesimcloze)
lmeplau = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + cloze + plausibility + (1|ID) + (1|Subject), data = FFD2)
summary(lmeplau)

# but in both cases, it seems that when including cloze as a predictor, it incorporates their variance


################ introduce Concreteness ###############
onlyConc = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + ConcM + (1|ID) + (1|Subject), data = FFD2)
summary(onlyConc)

anova(lmeBasic,onlyConc)

# Concreteness affects fixation durations, worth including it in the model

################# let's check sensorimotor strength
onlySM = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + mink3_SM + (1|ID) + (1|Subject), data = FFD2)
summary(onlySM)

# also sensorimotor strength affects fixation durations, but less variance explained

interaction_Conc.Cloze = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + cloze*ConcM + (1|ID) + (1|Subject), data = FFD2)
additive_Conc.Cloze = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + cloze + ConcM + (1|ID) + (1|Subject), data = FFD2)

summary(interaction_Conc.Cloze) 
summary(additive_Conc.Cloze) 
anova(interaction_Conc.Cloze,additive_Conc.Cloze)
# 

### cannot install brms, so using BayesFactor package
# full_brms = brm(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + ConcM + (1|ID) + (1|Subject),
#                 data = FFD2, save_all_pars = TRUE, iter = 10000)
# null_brms = update(full_brms, formula = ~ .-ConcM)  # Same but without the ConcM term
# BF_brms_bridge = bayes_factor(full_brms, null_brms)
# BF_brms_bridge

# Now let's look at Bayes Factors (JZS prior)

FFD2$ID = factor(FFD2$ID)  # BayesFactor wants the random to be a factor
FFD2$Subject = factor(FFD2$Subject)

full_BF = lmBF(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + ConcM + ID + Subject,
               data = FFD2, whichRandom = c('ID', 'Subject'))
null_BF = lmBF(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + ID + Subject,
               data = FFD2, whichRandom = c('ID', 'Subject'))
full_BF / null_BF
# Concreteness has BF =  5.618111 ±1.61% when included in the base model


full_BF = lmBF(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + cloze + ConcM + ID + Subject,
               data = FFD2, whichRandom = c('ID', 'Subject'))
null_BF = lmBF(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + cloze + ID + Subject,
               data = FFD2, whichRandom = c('ID', 'Subject'))
full_BF / null_BF
# Concreteness has BF =  0.6181144 ±0.94% when included in the model with cloze

confint(additive_Conc.Cloze)

# interestingly, effect of concreteness and (log-)frequency are of similar magnitude

# alternative visualisation
densityplot(profile(additive_Conc.Cloze))
densityplot(profile(interaction_Conc.Cloze))

########## EXPLORATORY ################
arousal = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + A_MeanSum + (1|ID) + (1|Subject), data = FFD2)
summary(arousal)
# no effect of arousal on ffd

valence = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + V_MeanSum + (1|ID) + (1|Subject), data = FFD2)
summary(valence)
# no effect of valence on ffd

similarity = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + similarity + (1|ID) + (1|Subject), data = FFD2)
summary(valence)
# similarity has an effect, but not when introducing also cloze

plau = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + plausibility + (1|ID) + (1|Subject), data = FFD2)
summary(plau)