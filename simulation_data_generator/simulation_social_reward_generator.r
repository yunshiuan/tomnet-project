library(dplyr)
library(e1071)
list_mu = seq(2,8,length.out = 3)
list_sd = seq(0.1,2.1,length.out = 3)
list_sk = seq(-1,1,length.out = 3)
df = as.data.frame(expand.grid(list_mu,list_sd,list_sk))
names(df) = c("mu","sd","sk")
n_agent = 27
n_target = 4
index = 1
while(index<n_agent){
  # create a new set and see if it satisfies
  candidate = runif(n = n_target,min = 0,max = 10)
  cand_mean = round(mean(candidate),0)
  cand_sd = round(sd(candidate),1)
  cand_sk = round(skewness(candidate),0)
  
  if(index<=nrow(df)){
    cand_mean = df[index,1]
    cand_sd = df[index,2]
    cand_sk = df[index,3]
    #save candidate
    result = as.data.frame(0.05307378*candidate,row.names = c("subj's family1","subj's family2","subj's family3","subj's family4"))
    filename = as.character(paste("/Users/elaine/Desktop/TOMNET/tomnet-project/simulation_data_generator/30agents/S00",index+3,"b.csv"))
    write.csv(result,filename,row.names = TRUE)
    index=index+1
   
  }
}

