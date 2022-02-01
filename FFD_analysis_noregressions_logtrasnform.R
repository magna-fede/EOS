# import stuff
library(readxl)
library(lme4)
library(lmerTest)
library(languageR)
library(lattice)
library(BayesFactor)

# import dataset
FFD2 <- read.csv('C:/Users/fm02/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_ffd_41.csv')
# consider only values greater than 0
# previously selected only fixations 80-600ms long
FFD2 <- FFD2[FFD2$ms != 0, ]

# consider only words that were not regressed
FFD2 <- FFD2[FFD2$regressed != 1, ]

colnames(FFD2)

FFD2$logms = log(FFD2$ms)

additive_Conc.Cloze = lmer(logms ~ LogFreqZipf + PRECEDING_LogFreqZipf + Position + cloze + ConcM + (1|ID) + (1|Subject), data = FFD2)

summary(additive_Conc.Cloze) 


# alternative visualisation
densityplot(profile(additive_Conc.Cloze))

qqnorm(residuals(additive_Conc.Cloze))
qqline(residuals(additive_Conc.Cloze))
sjPlot::tab_model(additive_Conc.Cloze)

####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################

# import dataset
GD2 <- read.csv('C:/Users/fm02/OwnCloud/EOS_EyeTrackingDataCollection/Data_Results/data_forR/norm_gd_41.csv')
# consider only values greater than 0
# previously selected only fixations 80-600ms long
GD2 <- GD2[GD2$ms != 0, ]
# GD2 <- GD2[!(GD2$Subject== 3 | GD2$Subject== 14 | GD2$Subject== 33 | GD2$Subject== 34 ),]


# consider only words that were not regressed
GD22 <- GD2[GD2$regressed != 1, ]

colnames(GD2)

GD2$logms = log(GD2$ms)

additive_ConcM.Sim = lmer(logms ~ LogFreqZipf + LEN + Sim + ConcM + (1|ID) + (1|Subject), data = GD2)

summary(additive_ConcM.Sim) 


# alternative visualisation
densityplot(profile(additive_ConcM.Sim))

qqnorm(residuals(additive_ConcM.Sim))
qqline(residuals(additive_ConcM.Sim))
sjPlot::tab_model(additive_ConcM.Sim)
