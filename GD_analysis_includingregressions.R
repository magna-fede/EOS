# this script analyses which factors affect GD including trials with regressions

# import stuff
library(readxl)
library(lme4)
library(lmerTest)
library(languageR)
library(lattice)
library(BayesFactor)
library(ggeffects)
library(ggplot2)

# import dataset
GD2 <- read.csv('C:/Users/fm02/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_gd_41_withSemDAoA.csv')
GD2 <- GD2[GD2$ms > 80, ]

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

################ introduce Predictability(a priori), cloze, and semanticpredictability(=sim) ###############
lmeOnlyPred = lmer(ms ~ LogFreqZipf + LEN + PRECEDING_LogFreqZipf + Position +  Predictability +
                     (1|ID) + (1|Subject), data = GD2)
summary(lmeOnlyPred)

lmeOnlyCloze = lmer(ms ~ LogFreqZipf + LEN + PRECEDING_LogFreqZipf +  Position + cloze +
                      (1|ID) + (1|Subject), data = GD2)
summary(lmeOnlyCloze)

lmeOnlySemSim = lmer(ms ~ LogFreqZipf + LEN + PRECEDING_LogFreqZipf +  Position + Sim +
                       (1|ID) + (1|Subject), data = GD2)
summary(lmeOnlySemSim)

anova(lmeBasicGD2,lmeOnlyPred)
anova(lmeBasicGD2, lmeOnlyCloze)
anova(lmeBasicGD2, lmeOnlySemSim)

# In this case, all are significant, but sim explains the most variance

################ introduce Concreteness ###############
onlyConc = lmer(ms ~ LogFreqZipf + LEN  + PRECEDING_LogFreqZipf + Position +  ConcM + (1|ID) + (1|Subject), data = GD2)
summary(onlyConc)
# concreteness does not affect GD
anova(lmeBasicGD2,onlyConc)
# effect of concreteness (only marginal)

################# let's check sensorimotor strength
onlySM = lmer(ms ~ LogFreqZipf + LEN + PRECEDING_LogFreqZipf + Position + mink3_SM + (1|ID) + (1|Subject), data = GD2)
anova(lmeBasicGD2,onlySM)
# also Sensorimotor strength does not have an effect on GD

additive_ConcM.Sim = lmer(ms ~ LogFreqZipf + LEN + PRECEDING_LogFreqZipf + Position + Sim + ConcM + (1|ID) + (1|Subject), data = GD2)
interaction_ConcM.Sim = lmer(ms ~ LogFreqZipf + LEN + PRECEDING_LogFreqZipf + Position +
                             + Sim * ConcM + (1|ID) + (1|Subject), data = GD2)

summary(additive_ConcM.Sim) 
summary(interaction_ConcM.Sim)

performance(additive_ConcM.Sim)
performance(interaction_ConcM.Sim)

# concreteness is not significant anymore when including clozeSemSim in the model
# somehow the two share some variance
# this code will plot your table of interest
sjPlot::tab_model(additive_ConcM.Sim)
sjPlot::plot_model(additive_ConcM.Sim)

sjPlot::tab_model(interaction_ConcM.Sim, pred.labels = c('Intercept',
                                                      'Frequency (Zipf)',
                                                      'Length',
                                                      'Preceding Frequency (Zipf)',
                                                      'Position',
                                                      'Predictability',
                                                      'Concreteness',
                                                      'Predictability*Concreteness'))

# plot both concreteness and Sim
dfConcSim <- ggpredict(additive_ConcM.Sim, terms = c("Sim", "ConcM"))
plot(dfConcSim)

# plot both Frequency and Sim
dfFreqSim <- ggpredict(additive_ConcM.Sim, terms = c("Sim", "LogFreqZipf"))
plot(dfFreqSim)


GD2$ID = factor(GD2$ID)  # BayesFactor wants the random to be a factor
GD2$Subject = factor(GD2$Subject)


full_BF_int = lmBF(ms ~ LogFreqZipf + LEN + PRECEDING_LogFreqZipf + Position + Sim + ConcM + Sim:ConcM + ID + Subject,
                   data = GD2, whichRandom = c('ID', 'Subject'))

############################################

chainsFull_int <- posterior(full_BF_int, iterations = 10000,columnFilter="^ID$")

summary(chainsFull_int[,c("LogFreqZipf",
                          "LEN",
                          "PRECEDING_LogFreqZipf",
                          "Position", 
                          "Sim",
                          "ConcM", 
                          "Sim.&.ConcM")])

mcmc_intervals(chainsFull_int[,c("LogFreqZipf",
                                 "LEN",
                                 "PRECEDING_LogFreqZipf",
                                 "Position", 
                                 "Sim",
                                 "ConcM", 
                                 "Sim.&.ConcM")],
               prob=.5,prob_outer = .9, point_est = "mean")

###################### EXPLORATORY ######################
interaction_sheikh = lmer(ms ~ LogFreqZipf*A_MeanSum*ConcM + LEN + PRECEDING_LogFreqZipf + Position +
                               + Sim  + (1|ID) + (1|Subject), data = GD2)
summary(interaction_sheikh)
performance(interaction_sheikh)

interaction_A = lmer(ms ~ LogFreqZipf + A_MeanSum + LEN + PRECEDING_LogFreqZipf + Position +
                            + Sim*ConcM  + (1|ID) + (1|Subject), data = GD2)
summary(interaction_A)

interaction_V = lmer(ms ~ LogFreqZipf + V_MeanSum + LEN + PRECEDING_LogFreqZipf + Position +
                       + Sim*ConcM  + (1|ID) + (1|Subject), data = GD2)
summary(interaction_A)
