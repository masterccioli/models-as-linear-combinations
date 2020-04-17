rm(list=ls())
library(effects)
library(car)

data = read.csv('data.csv')
#attach(data)
#attach(cigData)

cor(data[,c('first','second','glove','cbow','skipgram','lsa','pmi')])
scatterplotMatrix(data[,c('first','second','glove','cbow','skipgram','lsa','pmi')])

hist(data$first)
hist(data$second)
hist(data$glove)
hist(data$cbow)
hist(data$skipgram)
hist(data$lsa)
hist(abs(data$pmi))


lm_glove = lm( data$glove ~ 1 + data$first+ data$second)
lm_cbow = lm( data$cbow ~ 1 + data$first+ data$second)
lm_skipgram = lm( data$skipgram ~ 1 + data$first+ data$second)
lm_lsa = lm( data$lsa ~ 1 + data$first+ data$second)
lm_pmi = lm( data$pmi ~ 1 + data$first+ data$second)

summary(lm_glove)
summary(lm_cbow)
summary(lm_skipgram)
summary(lm_lsa)
summary(lm_pmi)

d_matrix = rbind(c(lm_glove$coefficients[1],lm_glove$coefficients[2],lm_glove$coefficients[3],summary(lm_glove)$r.squared),
      c(lm_cbow$coefficients[1],lm_cbow$coefficients[2],lm_cbow$coefficients[3],summary(lm_cbow)$r.squared),
      c(lm_skipgram$coefficients[1],lm_skipgram$coefficients[2],lm_skipgram$coefficients[3],summary(lm_skipgram)$r.squared),
      c(lm_lsa$coefficients[1],lm_lsa$coefficients[2],lm_lsa$coefficients[3],summary(lm_lsa)$r.squared),
      c(lm_pmi$coefficients[1],lm_pmi$coefficients[2],lm_pmi$coefficients[3],summary(lm_pmi)$r.squared))

write.csv(d_matrix,'lm_coefficients.csv')
confint(lm_f)
confint(lm_s)
confint(lm_sh_f)
confint(lm_r_f)

lm_all = lm( data$IRT ~ 1 + data$first_order + data$second_order + abs(data$delta_frequency_sheer) + abs(data$delta_frequency_rank))
summary(lm_all, corr = TRUE)
confint(lm_all)

# look at different measures of frequency
lm_all = lm( data$IRT ~ data$first_order + data$second_order + data$delta_frequency_sheer)
summary(lm_all, corr = TRUE)
lm_all = lm( data$IRT ~ data$first_order + data$second_order + data$delta_frequency_log)
summary(lm_all, corr = TRUE)
lm_all = lm( data$IRT ~ data$first_order + data$second_order + data$delta_frequency_rank)
summary(lm_all, corr = TRUE)

# look at abs of frequency
lm_all = lm( data$IRT ~ data$first_order + data$second_order + abs(data$delta_frequency_sheer))
summary(lm_all, corr = TRUE)
lm_all = lm( data$IRT ~ data$first_order + data$second_order + abs(data$delta_frequency_log))
summary(lm_all, corr = TRUE)
lm_all = lm( data$IRT ~ data$first_order + data$second_order + abs(data$delta_frequency_rank))
summary(lm_all, corr = TRUE)



lm_all = lm( IRT ~ 1 + first_order + second_order + delta_frequency_sheer + abs(delta_frequency_sheer))
summary(lm_all, corr = TRUE)
lm_all = lm( IRT ~ 1 + first_order + second_order + delta_frequency_sheer)
summary(lm_all, corr = TRUE)
lm_all = lm( IRT ~ 1 + first_order + second_order + delta_frequency_log)
summary(lm_all, corr = TRUE)
lm_all = lm( IRT ~ 1 + first_order + second_order + abs(delta_frequency_log))
summary(lm_all, corr = TRUE)
confint(lm_all)

layout(matrix(1:4, nrow = 2))
par(mar = c(3.5, 3.5, 2.5, 1), mgp = c(2.0,0.7,0))
plot( lm_all, which = c(1, 2, 4, 5))

# look at full interaction model
interactionForm = "IRT ~ first_order * second_order * delta_frequency_sheer * delta_frequency_rank"
InterceptOnly = "IRT ~ 1"

lmInteraction = lm(interactionForm)
nParMult = log(nrow(data))
stepBackInteraction = step(lmInteraction, scope = list(lower = InterceptOnly,
                                                       upper = interactionForm),
                           direction = "backward",
                           k = nParMult)

lm_best_interaction = lm(IRT ~ first_order + second_order + delta_frequency_sheer + delta_frequency_rank + 
                           first_order:second_order + delta_frequency_sheer:delta_frequency_rank)
summary(lm_best_interaction)

###############################################
Upper = "data$IRT ~ data$first_order * data$second_order + abs(data$delta_frequency_sheer)"
lm_all = lm( Upper)
Lower = "IRT ~ 1"
nParMult = log(nrow(data))
stepBackInteraction = step(lm_all, scope = list(lower = Lower,
                                                       upper = Upper),
                           direction = "backward",
                           k = nParMult)

#########################################################
# predict clusters
rm(list=ls())
library(effects)
library(car)
source("Kruschke-P553-Utilities.R")
# Graph logistic function
logistic = function( x , b0=0 , b1=1 ) {
  y = 1/(1+exp(-(b0+b1*x)))
  return( y )
}

data = read.csv('fluency_data_matrix_w3.csv')
cor(data[,c('first_order','second_order','delta_frequency_sheer','delta_frequency_rank','delta_frequency_log','IRT','switch')])




logm = glm( data$switch ~ data$IRT , data=data , 
                 family=binomial(link="logit") )
summary( logm )

# look at different measures of frequency
logm = glm( data$switch ~ data$first_order + data$second_order + data$delta_frequency_sheer, data=data , 
            family=binomial(link="logit") )
summary( logm )
logm = glm( data$switch ~ data$first_order + data$second_order + data$delta_frequency_rank, data=data , 
            family=binomial(link="logit") )
summary( logm )
logm = glm( data$switch ~ data$first_order + data$second_order + data$delta_frequency_log, data=data , 
            family=binomial(link="logit") )
summary( logm )

# look at abs of frequency
logm = glm( data$switch ~ data$first_order + data$second_order + abs(data$delta_frequency_sheer), data=data , 
            family=binomial(link="logit") )
summary( logm )
logm = glm( data$switch ~ data$first_order + data$second_order + abs(data$delta_frequency_rank), data=data , 
            family=binomial(link="logit") )
summary( logm )
logm = glm( data$switch ~ data$first_order + data$second_order + abs(data$delta_frequency_log), data=data , 
            family=binomial(link="logit") )
summary( logm )

# with IRTs as predictor
logm = glm( data$switch ~ data$first_order + data$second_order + abs(data$delta_frequency_sheer) + data$IRT, data=data , 
            family=binomial(link="logit") )
summary( logm )


# Make plot of data with logistic function:
b0=logm$coef[1]
b1=logm$coef[2]
thresh = -b0/b1
slope = b1*0.25
openGraph()
plot( jitter(as.numeric(data$switch),0.05) ~ data$IRT , data=data , 
      main=bquote( atop( "logistic(x), b0="*.(b0)*", b1="*.(b1) ,
                         "=> thresh="*.(thresh)*", slope="*.(slope) ) ) , 
      xlab="IRT" , ylab="p(y=1=)" )
xComb = with( data , seq(min(data$IRT), max(data$IRT),length=201) )
lines( xComb , logistic( xComb , b0=b0 , b1=b1 ) , col=1 , type="l" , lwd=2 )
abline(v=thresh,lty="dotted",col="green")
abline(h=0.5,lty="dotted",col="green")
abline(a=0.5-slope*thresh , b=slope , lty="dotted",col="red")

