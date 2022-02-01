library(sjPlot)
library(sjlabelled)
library(sjmisc)
library(ggplot2)
library(wesanderson)
# 
# plot_model(additive_Conc.Cloze)
# library(ggeffects)
# 
# # plot effect of ConcM
# prCM <- ggpredict(l, "ConcM")
# plot(prCM)
# 
# 
# # plot effect of cloze
# prcl <- ggpredict(additive_Conc.Cloze, "cloze")
# plot(prcl)
# 
# p <- plot_model(additive_Conc.Cloze, type = "eff", terms = "cloze", colors = "Set2") + aes(linetype=group, color=group)
# 
# plot_model(interaction_Conc.Cloze, type = "eff", terms = "cloze*ConcM")
# p <- plot_model(m)

##### PLOTs for FFD_noregressions

# plot only Concreteness
dfConc <- ggpredict(additive_Conc.Cloze, terms = c("ConcM"))
ggplot(dfConc, aes(x, predicted)) + 
  geom_line(aes(linetype=group, color=group)) +
  geom_ribbon(aes(ymin=conf.low, ymax=conf.high, fill=group), alpha=0.15) +
  scale_linetype_manual(values = c("solid")) +
  xlab("concreteness") +
  ylab("FFD") +
  ggtitle("Effects of concretess on FFD")

# plot only cloze
dfCloze <- ggpredict(additive_Conc.Cloze, terms = c("cloze"))
ggplot(dfCloze, aes(x, predicted)) + 
  geom_line(aes(linetype=group, color=group)) +
  geom_ribbon(aes(ymin=conf.low, ymax=conf.high, fill=group), alpha=0.15) +
  scale_linetype_manual(values = c("solid"))  +
  xlab("cloze probability") +
  ylab("FFD") +
  ggtitle("Effects of predictability on FFD")

# plot both concreteness and cloze
dfConcCloze <- ggpredict(additive_Conc.Cloze, terms = c("cloze", "ConcM"))
plot(dfConcCloze)

# plot both Frequency and cloze
dfFreqCloze <- ggpredict(additive_Conc.Cloze, terms = c("cloze", "LogFreqZipf"))
plot(dfFreqCloze)


# plot all points
ggplot(additive_Conc.Cloze,aes(y=ms,x=cloze,color=ConcM))+
  geom_point(size = 1)+
  stat_smooth(method="lm",se=FALSE)

# this is informative in that it suggests that it seems that data more variable in 
# low cloze sentences then in high cloze sentences


# this code will plot your table of interest
sjPlot::tab_model(additive_Conc.Cloze)
