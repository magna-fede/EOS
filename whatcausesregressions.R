# this script takes as input whether a word was followed by a regression or not
# We want to inspect which factors make a word more likely to be regressed.

# import stuff
library(readxl)
library(lme4)
library(lmerTest)
library(languageR)
library(lattice)
library(BayesFactor)

# import dataset
FFD3 <- read.csv('C:/Users/fm02/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/regressed_norminfo.csv')

# FFD3 <- FFD3[!(FFD3$Subject== 3 | FFD3$Subject== 14 | FFD3$Subject== 33 | FFD3$Subject== 34 ),]

# rescale trialnr
trialnr_scaled <- as.data.frame(scale(FFD3[,"trialnr"]))
FFD3$trialnrscaled=unlist(trialnr_scaled)

################ trial number (order) ###############
trialsFFD3 = glmer(ms ~ trialnrscaled + (1|ID) + (1|Subject), data = FFD3, family=binomial)
summary(trialsFFD3)
# the trial number affects the probability of making a regression

################ length preceding word ###############
lenPREC_FFD3 = glmer(ms ~ LENprec + (1|ID) + (1|Subject), data = FFD3, family=binomial)
summary(lenPREC_FFD3)
# the length of the preceding word affects the probability of making a regression

################ length word ###############
lenFFD3 = glmer(ms ~ LEN + (1|ID) + (1|Subject), data = FFD3, family=binomial)
summary(lenFFD3)
# length does not affects the probability of making a regression

################ zipf value of the word ###############
freqFFD3 = glmer(ms ~ LogFreqZipf + (1|ID) + (1|Subject), data = FFD3, family=binomial)
summary(freqFFD3)
# frequency marginally affects the probability of making a regression

################ time before first fixation on the word ###############
tbffdFFD3 = glmer(ms ~ time_before_ff + (1|ID) + (1|Subject), data = FFD3, family=binomial)
summary(tbffdFFD3)
# time before FFD affects the probability of making a regression

posFFD3 = glmer(ms ~ Position + (1|ID) + (1|Subject), data = FFD3, family=binomial)
summary(posFFD3)
# time before FFD affects the probability of making a regression

################ this is our basic model ###############
lmerBasic = glmer(ms ~ trialnrscaled + LENprec + Position + time_before_ff + (1|ID) + (1|Subject), data = FFD3, family=binomial)
summary(lmerBasic)
# not including Frequency

################ plausibility of the sentence ###############
plauFFD3 = glmer(ms ~ trialnrscaled + LENprec + Position + time_before_ff + plausibility + (1|ID) + (1|Subject), data = FFD3, family=binomial)
summary(plauFFD3)
# plausibility of the sentence affects the probability of making a regression

################ cloze probability ###############
clozeFFD3 = glmer(ms ~ trialnrscaled + LENprec + Position + time_before_ff + cloze + (1|ID) + (1|Subject), data = FFD3, family=binomial)
summary(clozeFFD3)
# cloze affects the probability of making a regression

################ cloze semantic similarity ###############
semSimFFD3 = glmer(ms ~ trialnrscaled + LENprec + Position + time_before_ff + Sim + (1|ID) + (1|Subject), data = FFD3, family=binomial)
summary(semSimFFD3)
# semantic similarity affects the probability of making a regression

################ check which best ###############
anova(lmerBasic,clozeFFD3)
anova(lmerBasic,semSimFFD3)
# In this case, all are significant, but cloze explains the most variance
# (though same order of magnitude).

################ concreteness ###############
concMFFD3 = glmer(ms ~ trialnrscaled + LENprec + Position + time_before_ff + ConcM + (1|ID) + (1|Subject), data = FFD3, family=binomial)
summary(concMFFD3)
# ConcM affects the probability of making a regression


confint(clozeFFD3)

# variables that affect probability of making a regression:
# trialnrscaled (less likely for later trials)
# LENprec (less likely for longer preceding words)
# Position (more likely at later positions) 
# time_before_ff (less likely if spent more time in the sentence)
# cloze (less likely for higher cloze)