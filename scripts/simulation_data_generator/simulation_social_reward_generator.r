library(dplyr)
library(e1071)
raw_target = read.csv(file = "/Users/elaine/Desktop/TOMNET/tomnet-project/scripts/simulation_data_generator/agent_raw.csv", header = TRUE)
raw_target = subset(raw_target,select=-c(X))
raw_target_score = subset(raw_target,select=c(A,B,C,D))
raw_target = cbind(raw_target,mu = apply(raw_target_score,1,mean),sd = apply(raw_target_score,1,sd),sk = apply(raw_target_score,1,skewness))
#list_mu = seq(2,8,length.out = 4)
#list_sd = seq(0.1,2.1,length.out = 3)
#list_sk = seq(-1,1,length.out = 3)
#df = as.data.frame(expand.grid(list_mu,list_sd,list_sk))
#names(df) = c("mu","sd","sk")
n_agent = 25
#n_target = 4
#NON-LINEAR
#y=function(x) {-x*log2(x)}
#MAX_S_R = optimise(y, c(0,1),maximum = TRUE)[["objective"]]



i = 1
while(i<=n_agent){
  # create a new set and see if it satisfies
  #candidate = runif(n = n_target,min = 0,max = 10)
  #cand_mean = round(mean(candidate),0)
  #cand_sd = round(sd(candidate),1)
  #cand_sk = round(skewness(candidate),0)
  
  #if(nrow(subset(df,mu == cand_mean & sd == cand_sd  & sk == cand_sk))>0){
    #save candidate
    #row_index = subset.data.frame(df,mu == cand_mean & sd == cand_sd  & sk == cand_sk)
    #names(row_index) = c("mean","SD","skewness")
    #result = as.data.frame(MAX_S_R*candidate/10,row.names = c("subj's family1","subj's family2","subj's family3","subj's family4"))
    result = rbind(raw_target[i,"A"],raw_target[i,"B"],raw_target[i,"C"],raw_target[i,"D"])
    result = as.data.frame(result,row.names = c("subj's family1","subj's family2","subj's family3","subj's family4"))
    mu = raw_target[i, "mu"]
    sd = raw_target[i, "sd"]
    sk = raw_target[i, "sk"]
    result = cbind(result,mu,sd,sk)
    filename = as.character(paste0("/Users/elaine/Desktop/TOMNET/tomnet-project/scripts/simulation_data_generator/agents_from_human/human",raw_target[i,"subj"],"b.csv"))
    write.csv(result,filename,row.names = TRUE)
    # update the data frame
    #df = subset(df,mu != cand_mean | sd != cand_sd  | sk != cand_sk)
    i=i+1
   
  #}
}

