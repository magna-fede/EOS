# this script analyses which factors affect FFD including trials with regressions

# import stuff
library(readxl)
library(lme4)
library(lmerTest)
library(languageR)
library(lattice)
library(BayesFactor)
library(ggeffects)
library(ggplot2)
library(performance)

# import dataset
FFD2 <- read.csv('C:/Users/fm02/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_ffd_41.csv')
# consider only values greater than 0
# previously selected only fixations 80-600ms long

FFD2 <- FFD2[FFD2$ms != 0, ]

# no need to select, already selected the correct data in 220228,
# while if using 220225 you need to uncomment row below 
# FFD2 <- subset(FFD2, ms>80 & ms<600)

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

# All are significant, but SemanticSimilarity probability explains the most variance.

################ check other confounds ###############

# but in both cases, it seems that when including cloze as a predictor, it incorporates their variance


################ introduce Concreteness ###############
onlyConc = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + ConcM + (1|ID) + (1|Subject), data = FFD2)
summary(onlyConc)

anova(lmeBasic,onlyConc)

# Concreteness affects fixation durations, worth including it in the model

################# let's check sensorimotor strength
onlySM = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + mink3_SM + (1|ID) + (1|Subject), data = FFD2)
anova(lmeBasic,onlySM)

# also sensorimotor strength affects fixation durations, but less variance explained
interaction_Conc.SemSim = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + Sim*ConcM + (1|ID) + (1|Subject), data = FFD2)
additive_Conc.SemSim = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + Sim + ConcM + (1|ID) + (1|Subject), data = FFD2)

interaction_Conc.SemSim_withplausibility = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + plausibility + Sim*ConcM + (1|ID) + (1|Subject), data = FFD2)
additive_Conc.SemSim_withplausibility = lmer(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + plausibility + Sim + ConcM + (1|ID) + (1|Subject), data = FFD2)

summary(interaction_Conc.SemSim) 
summary(additive_Conc.SemSim) 
anova(interaction_Conc.SemSim,additive_Conc.SemSim)
# 

performance(additive_Conc.SemSim)
performance(interaction_Conc.SemSim)

performance(interaction_Conc.SemSim_withplausibility)

### cannot install brms, so using BayesFactor package
# full_brms = brm(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + ConcM + (1|ID) + (1|Subject),
#                 data = FFD2, save_all_pars = TRUE, iter = 10000)
# null_brms = update(full_brms, formula = ~ .-ConcM)  # Same but without the ConcM term
# BF_brms_bridge = bayes_factor(full_brms, null_brms)
# BF_brms_bridge

# Now let's look at Bayes Factors (JZS prior)

FFD2$ID = factor(FFD2$ID)  # BayesFactor wants the random to be a factor
FFD2$Subject = factor(FFD2$Subject)

conc_BF = lmBF(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + ConcM + ID + Subject,
               data = FFD2, whichRandom = c('ID', 'Subject'))
basic_BF = lmBF(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + ID + Subject,
               data = FFD2, whichRandom = c('ID', 'Subject'))
conc_BF / basic_BF
# Concreteness has BF =  7.492556 ±1.83% when included in the base model

add_BF = lmBF(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + Sim + ConcM + ID + Subject,
                data = FFD2, whichRandom = c('ID', 'Subject'))
Sim_BF = lmBF(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + Sim + ID + Subject,
                data = FFD2, whichRandom = c('ID', 'Subject'))
add_BF / Sim_BF
# Concreteness has BF = 0.8579711 ±1.46% when included in the model with Sim

add_BF / conc_BF # PF Sim parameter
# Sim has BF = 845334.2 ±1.74% also in the model with concreteness.


full_BF_int = lmBF(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + plausibility + Sim + ConcM + Sim:ConcM + ID + Subject,
                   data = FFD2, whichRandom = c('ID', 'Subject'))

interactionBF = full_BF_int / add_BF
interactionBF

# interaction has a BF = 0.09660833 ±1.39%



########### check best model #########################
# bf = generalTestBF(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + plausibility + Position +  
#                      + Sim * ConcM + Subject + ID, data=FFD2, whichRandom=c('ID', 'Subject'), neverExclude=c('ID', 'Subject'))

### Save an object to a file
# saveRDS(bf, file = "U:/AnEyeOnSemantics/41analysis/BF_additive_all.rds")
### Restore the object
# bf = readRDS(file = "U:/AnEyeOnSemantics/41analysis/BF_additive_all.rds")
#head(bf, n=10)
# Bayes factor analysis
# --------------
#   [1] PRECEDING_LogFreqZipf + Sim + Subject + ID                                   : 1.005547e+292 ±0.69%
# [2] PRECEDING_LogFreqZipf + Position + Sim + Subject + ID                        : 8.499006e+291 ±0.82%
# [3] PRECEDING_LogFreqZipf + Position + Sim + ConcM + Subject + ID                : 6.714375e+291 ±0.61%
# [4] PRECEDING_LogFreqZipf + Sim + ConcM + Subject + ID                           : 6.689156e+291 ±0.6%
# [5] Sim + Subject + ID                                                           : 5.944403e+291 ±1.18%
# [6] PRECEDING_LogFreqZipf + plausibility + Position + Sim + ConcM + Subject + ID : 5.933368e+291 ±0.65%
# [7] PRECEDING_LogFreqZipf + plausibility + Sim + ConcM + Subject + ID            : 5.837369e+291 ±0.61%
# [8] Sim + ConcM + Subject + ID                                                   : 5.639672e+291 ±0.72%
# [9] plausibility + Sim + ConcM + Subject + ID                                    : 5.239752e+291 ±0.64%
# [10] LogFreqZipf + PRECEDING_LogFreqZipf + Position + Sim + Subject + ID         : 4.67311e+291  ±0.49%
# 
# Against denominator:
#   Intercept only 
# ---
#   Bayes factor type: BFlinearModel, JZS
# 
# bf = generalTestBF(ms ~ LogFreqZipf + PRECEDING_LogFreqZipf + plausibility + Position + 
#                      SemD * Sim * ConcM + Subject + ID, data=FFD2,
#                    whichRandom=c('ID', 'Subject'), neverExclude=c('ID', 'Subject'))
# 
# ### Save an object to a file
# # saveRDS(bf, file = "U:/AnEyeOnSemantics/41analysis/BF_withSemD.rds")
# ### Restore the object
# bf2 = readRDS(file = "U:/AnEyeOnSemantics/41analysis/BF_withSemD.rds")
# head(bf2, n=10)
# Bayes factor analysis
# --------------
# [1] PRECEDING_LogFreqZipf + Sim + Subject + ID                                   : 1.827781e+291 ±0.84%
# [2] PRECEDING_LogFreqZipf + Position + Sim + Subject + ID                        : 1.521895e+291 ±0.6%
# [3] PRECEDING_LogFreqZipf + Sim + ConcM + Subject + ID                           : 1.204448e+291 ±0.52%
# [4] PRECEDING_LogFreqZipf + Position + Sim + ConcM + Subject + ID                : 1.18689e+291  ±0.98%
# [5] PRECEDING_LogFreqZipf + plausibility + Sim + ConcM + Subject + ID            : 1.158824e+291 ±0.63%
# [6] Sim + Subject + ID                                                           : 1.15757e+291  ±1.18%
# [7] PRECEDING_LogFreqZipf + plausibility + Position + Sim + ConcM + Subject + ID : 1.108098e+291 ±0.49%
# [8] plausibility + Sim + ConcM + Subject + ID                                    : 1.08734e+291  ±0.64%
# [9] Sim + ConcM + Subject + ID                                                   : 1.083316e+291 ±1.26%
# [10] LogFreqZipf + PRECEDING_LogFreqZipf + Position + Sim + Subject + ID         : 9.482608e+290 ±0.55%
# 
# Against denominator:
#   Intercept only 
# ---
#   Bayes factor type: BFlinearModel, JZS
# 

############################################

chainsFull_int <- posterior(full_BF_int, iterations = 10000,columnFilter="^ID$")

summary(chainsFull_int[,c("PRECEDING_LogFreqZipf",
                          "Position", 
                          "LogFreqZipf",
                          "plausibility",
                          "Sim",
                          "ConcM", 
                          "Sim.&.ConcM")])



# mcmc_areas(chainsFull_int[,c("PRECEDING_LogFreqZipf",
#                              "Position",
#                              "LogFreqZipf",
#                              "Sim",
#                              "ConcM",
#                              "Sim.&.ConcM")],
#            prob=.8,prob_outer = .9, point_est = "mean")


mcmc_intervals(chainsFull_int[,c("PRECEDING_LogFreqZipf",
                                 "Position",
                                 "LogFreqZipf",
                                 "plausibility",
                                 "Sim",
                                 "ConcM",
                                 "Sim.&.ConcM")],
               prob=.5,prob_outer = .9, point_est = "mean")




# # plot only Concreteness
# dfConc <- ggeffect(additive_Conc.SemSim, terms = c("ConcM"))
# ggplot(dfConc, aes(x, predicted)) + 
#   geom_line(aes(linetype=group, color=group)) +
#   geom_ribbon(aes(ymin=conf.low, ymax=conf.high, fill=group), alpha=0.15) +
#   scale_linetype_manual(values = c("solid")) +
#   xlab("concreteness") +
#   ylab("FFD") +
#   ggtitle("Effects of concretess on FFD")

# # plot only Sim
# dfSim <- ggeffect(additive_Conc.SemSim, terms = c("Sim"))
# ggplot(dfSim, aes(x, predicted)) + 
#   geom_line(aes(linetype=group, color=group)) +
#   geom_ribbon(aes(ymin=conf.low, ymax=conf.high, fill=group), alpha=0.15) +
#   scale_linetype_manual(values = c("solid"))  +
#   xlab("cloze semantic similarity") +
#   ylab("FFD") +
#   ggtitle("Effects of predictability on FFD")
# 

# plot both concreteness and Sim
dfConcSim <- ggpredict(interaction_Conc.SemSim, terms = c("Sim", "ConcM"))
plot(dfConcSim) +
  labs(x="Predictability", colour="Concreteness") +
  scale_color_discrete(labels = c("-1 SD", "Mean", "+1 SD"))


# plot both Frequency and Sim
dfFreqSim <- ggeffect(interaction_Conc.SemSim, terms = c("Sim", "LogFreqZipf"))
plot(dfFreqSim) + 
  labs(x="Predictability", color='Frequency') +
  scale_color_discrete(labels = c("-1 SD", "Mean", "+1 SD"))



# # plot all points
# ggplot(additive_Conc.SemSim,aes(y=ms,x=Sim,color=ConcM))+
#   geom_point(size = 1)+
#   stat_smooth(method="lm",se=FALSE)

# this is informative in that it suggests that it seems that data more variable in 
# low Sim sentences then in high Sim sentences


# this code will plot your table of interest
sjPlot::tab_model(additive_Conc.SemSim)
sjPlot::plot_model(additive_Conc.SemSim)

sjPlot::tab_model(interaction_Conc.SemSim_withplausibility, pred.labels = c('Intercept',
                                                           'Frequency (Zipf)',
                                                           'Preceding Frequency (Zipf)',
                                                           'Position',
                                                           'Plausibility',
                                                           'Predictability',
                                                           'Concreteness',
                                                           'Predictability*Concreteness'))

sjPlot::plot_model(interaction_Conc.SemSim_withplausibility, axis.labels = c('Predictability*Concreteness',
                                                            'Concreteness',
                                                            'Predictability',
                                                            'Plausibility',
                                                            'Position',
                                                            'Preceding Frequency (Zipf)',
                                                            'Frequency (Zipf)',
                                                            'Intercept'
                                                           ))
sjPlot::tab_model(interaction_Conc.SemSim, pred.labels = c('Intercept',
                                                                            'Frequency (Zipf)',
                                                                            'Preceding Frequency (Zipf)',
                                                                            'Position',
                                                                            #'Plausibility',
                                                                            'Predictability',
                                                                            'Concreteness',
                                                                            'Predictability*Concreteness'))

sjPlot::plot_model(interaction_Conc.SemSim, axis.labels = c('Predictability*Concreteness',
                                                                             'Concreteness',
                                                                             'Predictability',
                                                                             #'Plausibility',
                                                                             'Position',
                                                                             'Preceding Frequency (Zipf)',
                                                                             'Frequency (Zipf)',
                                                                             'Intercept'
))

sjPlot::plot_model(interaction_Conc.SemSim)

densityplot(profile(additive_Conc.SemSim))


# exploratory
interaction_V_MeanSum = lmer(ms ~ V_MeanSum + LogFreqZipf + PRECEDING_LogFreqZipf + Position + plausibility + Sim*ConcM + (1|ID) + (1|Subject), data = FFD2)
summary(interaction_V_MeanSum)

interaction_A_MeanSum = lmer(ms ~ A_MeanSum + LogFreqZipf + PRECEDING_LogFreqZipf + Position + plausibility + Sim*ConcM + (1|ID) + (1|Subject), data = FFD2)
summary(interaction_A_MeanSum)
