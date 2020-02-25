library(dplyr)
library(e1071)
list_mu = seq(2,8,length.out = 4)
list_sd = seq(0.1,2.1,length.out = 3)
list_sk = seq(-1,1,length.out = 3)
df = as.data.frame(expand.grid(list_mu,list_sd,list_sk))
names(df) = c("mu","sd","sk")
n_agent = 36
n_target = 4
index = 1
y=function(x) {-x*log2(x)}
MAX_S_R = optimise(y, c(0,1),maximum = TRUE)[["objective"]]

while(index<n_agent){
  # create a new set and see if it satisfies
  candidate = runif(n = n_target,min = 0,max = 10)
  cand_mean = round(mean(candidate),0)
  cand_sd = round(sd(candidate),1)
  cand_sk = round(skewness(candidate),0)
  
  if(nrow(subset(df,mu == cand_mean & sd == cand_sd  & sk == cand_sk))>0){
    #save candidate
    row_index = subset.data.frame(df,mu == cand_mean & sd == cand_sd  & sk == cand_sk)
    names(row_index) = c("mean","SD","skewness")
    result = as.data.frame(MAX_S_R*candidate/10,row.names = c("subj's family1","subj's family2","subj's family3","subj's family4"))
    result = cbind(result,row_index)
    filename = as.character(paste0("/Users/elaine/Desktop/TOMNET/tomnet-project/simulation_data_generator/30agents/S",sprintf("%03d",index+3),"b.csv"))
    write.csv(result,filename,row.names = TRUE)
    #read.csv(filename)
    # update the data frame
    df = subset(df,mu != cand_mean | sd != cand_sd  | sk != cand_sk)
    index=index+1
   
  }
}

