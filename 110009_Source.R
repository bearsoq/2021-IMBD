install.packages("stringr")
install.packages("glmnet")
install.packages("useful")
install.packages("xgboost")
memory.limit(size=9999999999)
#----------訓練集----------#
data <- read.table("C:/Users/User/Desktop/train20210817v2.csv",header = TRUE,sep = ",",fill=TRUE)
dim(data)
head(data)
str(data)#查看資料結構(型態)
data1<-data[,-1]
summary(data1)
sum(is.na(data1))
head(data1)
#F_1~F_13正規化
for (i in c(1:13)){
  data1[,i] <- (data1[,i]-min(data1[,i]))/(max(data1[,i])-min(data1[,i]))
  names(data1)[i]<-str_c("normalization_F_",i)
}
head(data1[,1:13])

#將正規化過的跟原本的合併
data1<- cbind(data[,-1],data1[-14]);head(data1)

#----------特徵工程----------#
##F_1~F_13正規化之後做連續變數取指數&平方&立方
library(stringr)
# 例外處理data1中F_1等於0的情況 
for (i in c(1:98072)){
  if (data1[i,"F_1"] == 0){
    data1[i,"F_1"] <- 0 + 1
   }
}
Continuous_feature_engineering <- function(dataset){
  Var <- c("normalization_F_1","normalization_F_2","normalization_F_3","normalization_F_4",
           "normalization_F_5","normalization_F_6","normalization_F_7","normalization_F_8",
           "normalization_F_9","normalization_F_10","normalization_F_11","normalization_F_12",
           "normalization_F_13")
  Var1 <- names(data1[,1:13])
  Exp_Var <- str_c(Var,"exp",sep="_")
  Square_Var <- str_c(Var,"square",sep = "_")
  Root_Var <- str_c(Var1,"root",sep = "_") #用原本的F1-F13號開根
  ln_var <- str_c(Var1,"ln",sep = "_") #用原本的F1-F13取log以e為底
  Cube_Var <-str_c(Var,"cube",sep="_")
  sin_var <- str_c(Var1,"sin", sep = "_")
  cos_var <- str_c(Var1,"cos", sep = "_")
  reciprocal_var <- str_c(Var1, "reciprocal", sep = "_")
  exp_matrix<-exp(dataset[Var])
  square_matrix <- dataset[Var]**2
  root_matrix <- sqrt(dataset[Var1])  #用原本的F1-F13號開根
  ln_matrix <- log(dataset[Var1]) #用原本的F1-F13取log以e為底
  cube_matrix <- dataset[Var]**3
  sin_matrix <- sin(dataset[Var1])
  cos_matrix <- cos(dataset[Var1])
  reciprocal_matrix<-1/dataset[Var1]
  colnames(exp_matrix) <- Exp_Var
  colnames(square_matrix) <- Square_Var
  colnames(root_matrix) <- Root_Var
  colnames(ln_matrix) <- ln_var
  colnames(cube_matrix)<-Cube_Var
  colnames(sin_matrix) <- sin_var
  colnames(cos_matrix) <- cos_var
  colnames(reciprocal_matrix) <- reciprocal_var
  dataset <- cbind(dataset,exp_matrix, square_matrix, root_matrix, ln_matrix, cube_matrix, sin_matrix, cos_matrix, reciprocal_matrix)
  return(dataset)
}
data1<-Continuous_feature_engineering(data1)
dim(data1)
sum(is.na(data1))
head(data1)

#F_1對其他變數做交互作用
Continuous_F_1 <- function(dataset1){
  Var <- c( "F_2","F_3","F_4","F_5","F_6","F_7","F_8","F_9","F_10","F_11","F_12","F_13",
            "normalization_F_2","normalization_F_3","normalization_F_4","normalization_F_5",
            "normalization_F_6","normalization_F_7","normalization_F_8","normalization_F_9",
            "normalization_F_10","normalization_F_11","normalization_F_12","normalization_F_13",
            "normalization_F_2_exp","normalization_F_3_exp","normalization_F_4_exp","normalization_F_5_exp",
            "normalization_F_6_exp","normalization_F_7_exp", "normalization_F_8_exp","normalization_F_9_exp",
            "normalization_F_10_exp","normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",   
            "normalization_F_2_square","normalization_F_3_square","normalization_F_4_square","normalization_F_5_square",
            "normalization_F_6_square","normalization_F_7_square","normalization_F_8_square","normalization_F_9_square", 
            "normalization_F_10_square","normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
            "F_2_root","F_3_root","F_4_root","F_5_root",                 
            "F_6_root","F_7_root","F_8_root","F_9_root",
            "F_10_root","F_11_root","F_12_root","F_13_root",
            "F_2_ln","F_3_ln","F_4_ln","F_5_ln",
            "F_6_ln","F_7_ln","F_8_ln","F_9_ln",
            "F_10_ln","F_11_ln","F_12_ln","F_13_ln",                  
            "normalization_F_2_cube","normalization_F_3_cube","normalization_F_4_cube","normalization_F_5_cube",
            "normalization_F_6_cube","normalization_F_7_cube","normalization_F_8_cube","normalization_F_9_cube",   
            "normalization_F_10_cube","normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
            "F_2_sin","F_3_sin","F_4_sin","F_5_sin",
            "F_6_sin","F_7_sin","F_8_sin","F_9_sin",
            "F_10_sin","F_11_sin","F_12_sin","F_13_sin",
            "F_2_cos","F_3_cos","F_4_cos","F_5_cos","F_6_cos","F_7_cos",
            "F_8_cos","F_9_cos","F_10_cos","F_11_cos","F_12_cos","F_13_cos",                 
            "F_2_reciprocal","F_3_reciprocal","F_4_reciprocal","F_5_reciprocal",
            "F_6_reciprocal","F_7_reciprocal","F_8_reciprocal","F_9_reciprocal",
            "F_10_reciprocal","F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
            "F_1","normalization_F_1","normalization_F_1_exp","normalization_F_1_square","F_1_root","F_1_ln",
            "normalization_F_1_cube","F_1_sin","F_1_cos", "F_1_reciprocal"
            )
  F_1 <- str_c("F_1",Var[1:120],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:120){
    B<-dataset1["F_1"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:121]
  colnames(A) <- F_1
  
  normalization_F_1 <- str_c("normalization_F_1",Var[1:121],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:121){
    bb<-dataset1["normalization_F_1"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:122]
  colnames(aa) <- normalization_F_1
  
  normalization_F_1_exp <- str_c("normalization_F_1_exp",Var[1:122],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:122){
    BB<-dataset1["normalization_F_1_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:123]
  colnames(AA) <- normalization_F_1_exp
  
  normalization_F_1_square<- str_c("normalization_F_1_square",Var[1:123],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:123){
    DD<-dataset1["normalization_F_1_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:124]
  colnames(D) <- normalization_F_1_square
  
  F_1_root <- str_c("F_1_root",Var[1:124],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:124){
    dd<-dataset1["F_1_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:125]
  colnames(cc) <- F_1_root
  
  F_1_ln <- str_c("F_1_ln",Var[1:125],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:125){
    ee<-dataset1["F_1_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:126]
  colnames(e) <-F_1_ln
  
  normalization_F_1_cube <- str_c("normalization_F_1_cube",Var[1:126],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:126){
    ff<-dataset1["normalization_F_1_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:127]
  colnames(f) <-normalization_F_1_cube
  
  F_1_sin <- str_c("F_1_sin",Var[1:127],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:127){
    gg<-dataset1["F_1_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:128]
  colnames(g) <-F_1_sin
  
  F_1_cos<- str_c("F_1_cos",Var[1:128],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:128){
    hh<-dataset1["F_1_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:129]
  colnames(h) <-F_1_cos
  
  F_1_reciprocal<- str_c("F_1_reciprocal",Var[1:129],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:129){
    ii<-dataset1["F_1_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:130]
  colnames(i) <-F_1_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
F1<-Continuous_F_1(data1)
dim(F1)

#F_2對其他變數做交互作用
Continuous_F_2 <- function(dataset1){
  Var <- c( "F_3","F_4","F_5","F_6","F_7","F_8","F_9","F_10","F_11","F_12","F_13",
            "normalization_F_3","normalization_F_4","normalization_F_5",
            "normalization_F_6","normalization_F_7","normalization_F_8","normalization_F_9",
            "normalization_F_10","normalization_F_11","normalization_F_12","normalization_F_13",
            "normalization_F_3_exp","normalization_F_4_exp","normalization_F_5_exp",
            "normalization_F_6_exp","normalization_F_7_exp", "normalization_F_8_exp","normalization_F_9_exp",
            "normalization_F_10_exp","normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",   
            "normalization_F_3_square","normalization_F_4_square","normalization_F_5_square",
            "normalization_F_6_square","normalization_F_7_square","normalization_F_8_square","normalization_F_9_square", 
            "normalization_F_10_square","normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
            "F_3_root","F_4_root","F_5_root",                 
            "F_6_root","F_7_root","F_8_root","F_9_root",
            "F_10_root","F_11_root","F_12_root","F_13_root",
            "F_3_ln","F_4_ln","F_5_ln",
            "F_6_ln","F_7_ln","F_8_ln","F_9_ln",
            "F_10_ln","F_11_ln","F_12_ln","F_13_ln",                  
            "normalization_F_3_cube","normalization_F_4_cube","normalization_F_5_cube",
            "normalization_F_6_cube","normalization_F_7_cube","normalization_F_8_cube","normalization_F_9_cube",   
            "normalization_F_10_cube","normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
            "F_3_sin","F_4_sin","F_5_sin",
            "F_6_sin","F_7_sin","F_8_sin","F_9_sin",
            "F_10_sin","F_11_sin","F_12_sin","F_13_sin",
            "F_3_cos","F_4_cos","F_5_cos","F_6_cos","F_7_cos",
            "F_8_cos","F_9_cos","F_10_cos","F_11_cos","F_12_cos","F_13_cos",                 
            "F_3_reciprocal","F_4_reciprocal","F_5_reciprocal",
            "F_6_reciprocal","F_7_reciprocal","F_8_reciprocal","F_9_reciprocal",
            "F_10_reciprocal","F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
            "F_2","normalization_F_2","normalization_F_2_exp","normalization_F_2_square","F_2_root","F_2_ln",
            "normalization_F_2_cube","F_2_sin","F_2_cos", "F_2_reciprocal"
  )
  F_2 <- str_c("F_2",Var[1:110],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:110){
    B<-dataset1["F_2"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:111]
  colnames(A) <- F_2
  
  normalization_F_2 <- str_c("normalization_F_2",Var[1:111],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:111){
    bb<-dataset1["normalization_F_2"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:112]
  colnames(aa) <- normalization_F_2
  
  normalization_F_2_exp <- str_c("normalization_F_2_exp",Var[1:112],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:112){
    BB<-dataset1["normalization_F_2_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:113]
  colnames(AA) <- normalization_F_2_exp
  
  normalization_F_2_square<- str_c("normalization_F_2_square",Var[1:113],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:113){
    DD<-dataset1["normalization_F_2_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:114]
  colnames(D) <- normalization_F_2_square
  
  F_2_root <- str_c("F_2_root",Var[1:114],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:114){
    dd<-dataset1["F_2_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:115]
  colnames(cc) <- F_2_root
  
  F_2_ln <- str_c("F_2_ln",Var[1:115],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:115){
    ee<-dataset1["F_2_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:116]
  colnames(e) <-F_2_ln
  
  normalization_F_2_cube <- str_c("normalization_F_2_cube",Var[1:116],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:116){
    ff<-dataset1["normalization_F_2_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:117]
  colnames(f) <-normalization_F_2_cube
  
  F_2_sin <- str_c("F_2_sin",Var[1:117],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:117){
    gg<-dataset1["F_2_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:118]
  colnames(g) <-F_2_sin
  
  F_2_cos<- str_c("F_2_cos",Var[1:118],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:118){
    hh<-dataset1["F_2_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:119]
  colnames(h) <-F_2_cos
  
  F_2_reciprocal<- str_c("F_2_reciprocal",Var[1:119],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:119){
    ii<-dataset1["F_2_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:120]
  colnames(i) <-F_2_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
F2<-Continuous_F_2(data1)
dim(F2)

#F_3對其他變數做交互作用
Continuous_F_3 <- function(dataset1){
  Var <- c( "F_4","F_5","F_6","F_7","F_8","F_9","F_10","F_11","F_12","F_13",
            "normalization_F_4","normalization_F_5",
            "normalization_F_6","normalization_F_7","normalization_F_8","normalization_F_9",
            "normalization_F_10","normalization_F_11","normalization_F_12","normalization_F_13",
            "normalization_F_4_exp","normalization_F_5_exp",
            "normalization_F_6_exp","normalization_F_7_exp", "normalization_F_8_exp","normalization_F_9_exp",
            "normalization_F_10_exp","normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",   
            "normalization_F_4_square","normalization_F_5_square",
            "normalization_F_6_square","normalization_F_7_square","normalization_F_8_square","normalization_F_9_square", 
            "normalization_F_10_square","normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
            "F_4_root","F_5_root",                 
            "F_6_root","F_7_root","F_8_root","F_9_root",
            "F_10_root","F_11_root","F_12_root","F_13_root",
            "F_4_ln","F_5_ln",
            "F_6_ln","F_7_ln","F_8_ln","F_9_ln",
            "F_10_ln","F_11_ln","F_12_ln","F_13_ln",                  
            "normalization_F_4_cube","normalization_F_5_cube",
            "normalization_F_6_cube","normalization_F_7_cube","normalization_F_8_cube","normalization_F_9_cube",   
            "normalization_F_10_cube","normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
            "F_4_sin","F_5_sin",
            "F_6_sin","F_7_sin","F_8_sin","F_9_sin",
            "F_10_sin","F_11_sin","F_12_sin","F_13_sin",
            "F_4_cos","F_5_cos","F_6_cos","F_7_cos",
            "F_8_cos","F_9_cos","F_10_cos","F_11_cos","F_12_cos","F_13_cos",                 
            "F_4_reciprocal","F_5_reciprocal",
            "F_6_reciprocal","F_7_reciprocal","F_8_reciprocal","F_9_reciprocal",
            "F_10_reciprocal","F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
            "F_3","normalization_F_3","normalization_F_3_exp","normalization_F_3_square","F_3_root","F_3_ln",
            "normalization_F_3_cube","F_3_sin","F_3_cos", "F_3_reciprocal"
  )
  F_3 <- str_c("F_3",Var[1:100],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:100){
    B<-dataset1["F_3"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:101]
  colnames(A) <- F_3
  
  normalization_F_3 <- str_c("normalization_F_3",Var[1:101],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:101){
    bb<-dataset1["normalization_F_3"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:102]
  colnames(aa) <- normalization_F_3
  
  normalization_F_3_exp <- str_c("normalization_F_3_exp",Var[1:102],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:102){
    BB<-dataset1["normalization_F_3_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:103]
  colnames(AA) <- normalization_F_3_exp
  
  normalization_F_3_square<- str_c("normalization_F_3_square",Var[1:103],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:103){
    DD<-dataset1["normalization_F_3_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:104]
  colnames(D) <- normalization_F_3_square
  
  F_3_root <- str_c("F_3_root",Var[1:104],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:104){
    dd<-dataset1["F_3_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:105]
  colnames(cc) <- F_3_root
  
  F_3_ln <- str_c("F_3_ln",Var[1:105],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:105){
    ee<-dataset1["F_3_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:106]
  colnames(e) <-F_3_ln
  
  normalization_F_3_cube <- str_c("normalization_F_3_cube",Var[1:106],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:106){
    ff<-dataset1["normalization_F_3_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:107]
  colnames(f) <-normalization_F_3_cube
  
  F_3_sin <- str_c("F_3_sin",Var[1:107],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:107){
    gg<-dataset1["F_3_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:108]
  colnames(g) <-F_3_sin
  
  F_3_cos<- str_c("F_3_cos",Var[1:108],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:108){
    hh<-dataset1["F_3_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:109]
  colnames(h) <-F_3_cos
  
  F_3_reciprocal<- str_c("F_3_reciprocal",Var[1:109],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:109){
    ii<-dataset1["F_3_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:110]
  colnames(i) <-F_3_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
F3<-Continuous_F_3(data1)
dim(F3)

#F_4對其他變數做交互作用
Continuous_F_4 <- function(dataset1){
  Var <- c( "F_5","F_6","F_7","F_8","F_9","F_10","F_11","F_12","F_13",
            "normalization_F_5",
            "normalization_F_6","normalization_F_7","normalization_F_8","normalization_F_9",
            "normalization_F_10","normalization_F_11","normalization_F_12","normalization_F_13",
            "normalization_F_5_exp",
            "normalization_F_6_exp","normalization_F_7_exp", "normalization_F_8_exp","normalization_F_9_exp",
            "normalization_F_10_exp","normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",   
            "normalization_F_5_square",
            "normalization_F_6_square","normalization_F_7_square","normalization_F_8_square","normalization_F_9_square", 
            "normalization_F_10_square","normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
            "F_5_root",                 
            "F_6_root","F_7_root","F_8_root","F_9_root",
            "F_10_root","F_11_root","F_12_root","F_13_root",
            "F_5_ln",
            "F_6_ln","F_7_ln","F_8_ln","F_9_ln",
            "F_10_ln","F_11_ln","F_12_ln","F_13_ln",                  
            "normalization_F_5_cube",
            "normalization_F_6_cube","normalization_F_7_cube","normalization_F_8_cube","normalization_F_9_cube",   
            "normalization_F_10_cube","normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
            "F_5_sin",
            "F_6_sin","F_7_sin","F_8_sin","F_9_sin",
            "F_10_sin","F_11_sin","F_12_sin","F_13_sin",
            "F_5_cos","F_6_cos","F_7_cos",
            "F_8_cos","F_9_cos","F_10_cos","F_11_cos","F_12_cos","F_13_cos",                 
            "F_5_reciprocal",
            "F_6_reciprocal","F_7_reciprocal","F_8_reciprocal","F_9_reciprocal",
            "F_10_reciprocal","F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
            "F_4","normalization_F_4","normalization_F_4_exp","normalization_F_4_square","F_4_root","F_4_ln",
            "normalization_F_4_cube","F_4_sin","F_4_cos", "F_4_reciprocal"
  )
  F_4 <- str_c("F_4",Var[1:90],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:90){
    B<-dataset1["F_4"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:91]
  colnames(A) <- F_4
  
  normalization_F_4 <- str_c("normalization_F_4",Var[1:91],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:91){
    bb<-dataset1["normalization_F_4"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:92]
  colnames(aa) <- normalization_F_4
  
  normalization_F_4_exp <- str_c("normalization_F_4_exp",Var[1:92],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:92){
    BB<-dataset1["normalization_F_4_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:93]
  colnames(AA) <- normalization_F_4_exp
  
  normalization_F_4_square<- str_c("normalization_F_4_square",Var[1:93],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:93){
    DD<-dataset1["normalization_F_4_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:94]
  colnames(D) <- normalization_F_4_square
  
  F_4_root <- str_c("F_4_root",Var[1:94],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:94){
    dd<-dataset1["F_4_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:95]
  colnames(cc) <- F_4_root
  
  F_4_ln <- str_c("F_4_ln",Var[1:95],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:95){
    ee<-dataset1["F_4_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:96]
  colnames(e) <-F_4_ln
  
  normalization_F_4_cube <- str_c("normalization_F_4_cube",Var[1:96],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:96){
    ff<-dataset1["normalization_F_4_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:97]
  colnames(f) <-normalization_F_4_cube
  
  F_4_sin <- str_c("F_4_sin",Var[1:97],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:97){
    gg<-dataset1["F_4_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:98]
  colnames(g) <-F_4_sin
  
  F_4_cos<- str_c("F_4_cos",Var[1:98],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:98){
    hh<-dataset1["F_4_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:99]
  colnames(h) <-F_4_cos
  
  F_4_reciprocal<- str_c("F_4_reciprocal",Var[1:99],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:99){
    ii<-dataset1["F_4_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:100]
  colnames(i) <-F_4_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
F4<-Continuous_F_4(data1)
dim(F4)

#F_5對其他變數做交互作用
Continuous_F_5 <- function(dataset1){
  Var <- c("F_6","F_7","F_8","F_9","F_10","F_11","F_12","F_13",
           "normalization_F_6","normalization_F_7","normalization_F_8","normalization_F_9",
           "normalization_F_10","normalization_F_11","normalization_F_12","normalization_F_13",
           "normalization_F_6_exp","normalization_F_7_exp", "normalization_F_8_exp","normalization_F_9_exp",
           "normalization_F_10_exp","normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",
           "normalization_F_6_square","normalization_F_7_square","normalization_F_8_square","normalization_F_9_square", 
           "normalization_F_10_square","normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
           "F_6_root","F_7_root","F_8_root","F_9_root",
           "F_10_root","F_11_root","F_12_root","F_13_root",
           "F_6_ln","F_7_ln","F_8_ln","F_9_ln",
           "F_10_ln","F_11_ln","F_12_ln","F_13_ln",
           "normalization_F_6_cube","normalization_F_7_cube","normalization_F_8_cube","normalization_F_9_cube",   
           "normalization_F_10_cube","normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
           "F_6_sin","F_7_sin","F_8_sin","F_9_sin",
           "F_10_sin","F_11_sin","F_12_sin","F_13_sin",
           "F_6_cos","F_7_cos","F_8_cos","F_9_cos","F_10_cos","F_11_cos","F_12_cos","F_13_cos",                 
           "F_6_reciprocal","F_7_reciprocal","F_8_reciprocal","F_9_reciprocal",
           "F_10_reciprocal","F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
           "F_5","normalization_F_5","normalization_F_5_exp","normalization_F_5_square","F_5_root","F_5_ln",
           "normalization_F_5_cube","F_5_sin","F_5_cos", "F_5_reciprocal"
  )
  F_5 <- str_c("F_5",Var[1:80],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:80){
    B<-dataset1["F_5"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:81]
  colnames(A) <- F_5
  
  normalization_F_5 <- str_c("normalization_F_5",Var[1:81],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:81){
    bb<-dataset1["normalization_F_5"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:82]
  colnames(aa) <- normalization_F_5
  
  normalization_F_5_exp <- str_c("normalization_F_5_exp",Var[1:82],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:82){
    BB<-dataset1["normalization_F_5_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:83]
  colnames(AA) <- normalization_F_5_exp
  
  normalization_F_5_square<- str_c("normalization_F_5_square",Var[1:83],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:83){
    DD<-dataset1["normalization_F_5_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:84]
  colnames(D) <- normalization_F_5_square
  
  F_5_root <- str_c("F_5_root",Var[1:84],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:84){
    dd<-dataset1["F_5_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:85]
  colnames(cc) <- F_5_root
  
  F_5_ln <- str_c("F_5_ln",Var[1:85],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:85){
    ee<-dataset1["F_5_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:86]
  colnames(e) <-F_5_ln
  
  normalization_F_5_cube <- str_c("normalization_F_5_cube",Var[1:86],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:86){
    ff<-dataset1["normalization_F_5_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:87]
  colnames(f) <-normalization_F_5_cube
  
  F_5_sin <- str_c("F_5_sin",Var[1:87],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:87){
    gg<-dataset1["F_5_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:88]
  colnames(g) <-F_5_sin
  
  F_5_cos<- str_c("F_5_cos",Var[1:88],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:88){
    hh<-dataset1["F_5_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:89]
  colnames(h) <-F_5_cos
  
  F_5_reciprocal<- str_c("F_5_reciprocal",Var[1:89],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:89){
    ii<-dataset1["F_5_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:90]
  colnames(i) <-F_5_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
F5<-Continuous_F_5(data1)
dim(F5)

Continuous_F_6 <- function(dataset1){
  Var <- c("F_7","F_8","F_9","F_10","F_11","F_12","F_13",
           "normalization_F_7","normalization_F_8","normalization_F_9",
           "normalization_F_10","normalization_F_11","normalization_F_12","normalization_F_13",
           "normalization_F_7_exp", "normalization_F_8_exp","normalization_F_9_exp",
           "normalization_F_10_exp","normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",
           "normalization_F_7_square","normalization_F_8_square","normalization_F_9_square", 
           "normalization_F_10_square","normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
           "F_7_root","F_8_root","F_9_root",
           "F_10_root","F_11_root","F_12_root","F_13_root",
           "F_7_ln","F_8_ln","F_9_ln",
           "F_10_ln","F_11_ln","F_12_ln","F_13_ln",
           "normalization_F_7_cube","normalization_F_8_cube","normalization_F_9_cube",   
           "normalization_F_10_cube","normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
           "F_7_sin","F_8_sin","F_9_sin",
           "F_10_sin","F_11_sin","F_12_sin","F_13_sin",
           "F_7_cos","F_8_cos","F_9_cos","F_10_cos","F_11_cos","F_12_cos","F_13_cos",                 
           "F_7_reciprocal","F_8_reciprocal","F_9_reciprocal",
           "F_10_reciprocal","F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
           "F_6","normalization_F_6","normalization_F_6_exp","normalization_F_6_square","F_6_root","F_6_ln",
           "normalization_F_6_cube","F_6_sin","F_6_cos", "F_6_reciprocal"
  )
  F_6 <- str_c("F_6",Var[1:70],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:70){
    B<-dataset1["F_6"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:71]
  colnames(A) <- F_6
  
  normalization_F_6 <- str_c("normalization_F_6",Var[1:71],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:71){
    bb<-dataset1["normalization_F_6"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:72]
  colnames(aa) <- normalization_F_6
  
  normalization_F_6_exp <- str_c("normalization_F_6_exp",Var[1:72],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:72){
    BB<-dataset1["normalization_F_6_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:73]
  colnames(AA) <- normalization_F_6_exp
  
  normalization_F_6_square<- str_c("normalization_F_6_square",Var[1:73],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:73){
    DD<-dataset1["normalization_F_6_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:74]
  colnames(D) <- normalization_F_6_square
  
  F_6_root <- str_c("F_6_root",Var[1:74],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:74){
    dd<-dataset1["F_6_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:75]
  colnames(cc) <- F_6_root
  
  F_6_ln <- str_c("F_6_ln",Var[1:75],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:75){
    ee<-dataset1["F_6_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:76]
  colnames(e) <-F_6_ln
  
  normalization_F_6_cube <- str_c("normalization_F_6_cube",Var[1:76],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:76){
    ff<-dataset1["normalization_F_6_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:77]
  colnames(f) <-normalization_F_6_cube
  
  F_6_sin <- str_c("F_6_sin",Var[1:77],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:77){
    gg<-dataset1["F_6_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:78]
  colnames(g) <-F_6_sin
  
  F_6_cos<- str_c("F_6_cos",Var[1:78],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:78){
    hh<-dataset1["F_6_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:79]
  colnames(h) <-F_6_cos
  
  F_6_reciprocal<- str_c("F_6_reciprocal",Var[1:79],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:79){
    ii<-dataset1["F_6_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:80]
  colnames(i) <-F_6_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
F6<-Continuous_F_6(data1)
dim(F6)

#F_7對其他變數做交互作用
Continuous_F_7 <- function(dataset1){
  Var <- c("F_8","F_9","F_10","F_11","F_12","F_13",
           "normalization_F_8","normalization_F_9",
           "normalization_F_10","normalization_F_11","normalization_F_12","normalization_F_13",
           "normalization_F_8_exp","normalization_F_9_exp",
           "normalization_F_10_exp","normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",
           "normalization_F_8_square","normalization_F_9_square", 
           "normalization_F_10_square","normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
           "F_8_root","F_9_root",
           "F_10_root","F_11_root","F_12_root","F_13_root",
           "F_8_ln","F_9_ln",
           "F_10_ln","F_11_ln","F_12_ln","F_13_ln",
           "normalization_F_8_cube","normalization_F_9_cube",   
           "normalization_F_10_cube","normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
           "F_8_sin","F_9_sin",
           "F_10_sin","F_11_sin","F_12_sin","F_13_sin",
           "F_8_cos","F_9_cos","F_10_cos","F_11_cos","F_12_cos","F_13_cos",                 
           "F_8_reciprocal","F_9_reciprocal",
           "F_10_reciprocal","F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
           "F_7","normalization_F_7","normalization_F_7_exp","normalization_F_7_square","F_7_root","F_7_ln",
           "normalization_F_7_cube","F_7_sin","F_7_cos", "F_7_reciprocal"
  )
  F_7 <- str_c("F_7",Var[1:60],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:60){
    B<-dataset1["F_7"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:61]
  colnames(A) <- F_7
  
  normalization_F_7 <- str_c("normalization_F_7",Var[1:61],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:61){
    bb<-dataset1["normalization_F_7"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:62]
  colnames(aa) <- normalization_F_7
  
  normalization_F_7_exp <- str_c("normalization_F_7_exp",Var[1:62],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:62){
    BB<-dataset1["normalization_F_7_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:63]
  colnames(AA) <- normalization_F_7_exp
  
  normalization_F_7_square<- str_c("normalization_F_7_square",Var[1:63],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:63){
    DD<-dataset1["normalization_F_7_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:64]
  colnames(D) <- normalization_F_7_square
  
  F_7_root <- str_c("F_7_root",Var[1:64],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:64){
    dd<-dataset1["F_7_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:65]
  colnames(cc) <- F_7_root
  
  F_7_ln <- str_c("F_7_ln",Var[1:65],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:65){
    ee<-dataset1["F_7_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:66]
  colnames(e) <-F_7_ln
  
  normalization_F_7_cube <- str_c("normalization_F_7_cube",Var[1:66],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:66){
    ff<-dataset1["normalization_F_7_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:67]
  colnames(f) <-normalization_F_7_cube
  
  F_7_sin <- str_c("F_7_sin",Var[1:67],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:67){
    gg<-dataset1["F_7_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:68]
  colnames(g) <-F_7_sin
  
  F_7_cos<- str_c("F_7_cos",Var[1:68],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:68){
    hh<-dataset1["F_7_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:69]
  colnames(h) <-F_7_cos
  
  F_7_reciprocal<- str_c("F_7_reciprocal",Var[1:69],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:69){
    ii<-dataset1["F_7_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:70]
  colnames(i) <-F_7_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
F7<-Continuous_F_7(data1)
dim(F7)

#F_8對其他變數做交互作用
Continuous_F_8 <- function(dataset1){
  Var <- c("F_9","F_10","F_11","F_12","F_13",
           "normalization_F_9",
           "normalization_F_10","normalization_F_11","normalization_F_12","normalization_F_13",
           "normalization_F_9_exp",
           "normalization_F_10_exp","normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",
           "normalization_F_9_square", 
           "normalization_F_10_square","normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
           "F_9_root",
           "F_10_root","F_11_root","F_12_root","F_13_root",
           "F_9_ln",
           "F_10_ln","F_11_ln","F_12_ln","F_13_ln",
           "normalization_F_9_cube",   
           "normalization_F_10_cube","normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
           "F_9_sin",
           "F_10_sin","F_11_sin","F_12_sin","F_13_sin",
           "F_9_cos","F_10_cos","F_11_cos","F_12_cos","F_13_cos",                 
           "F_9_reciprocal",
           "F_10_reciprocal","F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
           "F_8","normalization_F_8","normalization_F_8_exp","normalization_F_8_square","F_8_root","F_8_ln",
           "normalization_F_8_cube","F_8_sin","F_8_cos", "F_8_reciprocal"
  )
  F_8 <- str_c("F_8",Var[1:50],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:50){
    B<-dataset1["F_8"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:51]
  colnames(A) <- F_8
  
  normalization_F_8 <- str_c("normalization_F_8",Var[1:51],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:51){
    bb<-dataset1["normalization_F_8"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:52]
  colnames(aa) <- normalization_F_8
  
  normalization_F_8_exp <- str_c("normalization_F_8_exp",Var[1:52],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:52){
    BB<-dataset1["normalization_F_8_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:53]
  colnames(AA) <- normalization_F_8_exp
  
  normalization_F_8_square<- str_c("normalization_F_8_square",Var[1:53],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:53){
    DD<-dataset1["normalization_F_8_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:54]
  colnames(D) <- normalization_F_8_square
  
  F_8_root <- str_c("F_8_root",Var[1:54],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:54){
    dd<-dataset1["F_8_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:55]
  colnames(cc) <- F_8_root
  
  F_8_ln <- str_c("F_8_ln",Var[1:55],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:55){
    ee<-dataset1["F_8_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:56]
  colnames(e) <-F_8_ln
  
  normalization_F_8_cube <- str_c("normalization_F_8_cube",Var[1:56],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:56){
    ff<-dataset1["normalization_F_8_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:57]
  colnames(f) <-normalization_F_8_cube
  
  F_8_sin <- str_c("F_8_sin",Var[1:57],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:57){
    gg<-dataset1["F_8_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:58]
  colnames(g) <-F_8_sin
  
  F_8_cos<- str_c("F_8_cos",Var[1:58],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:58){
    hh<-dataset1["F_8_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:59]
  colnames(h) <-F_8_cos
  
  F_8_reciprocal<- str_c("F_8_reciprocal",Var[1:59],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:59){
    ii<-dataset1["F_8_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:60]
  colnames(i) <-F_8_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
F8<-Continuous_F_8(data1)
dim(F8)

#F_9對其他變數做交互作用
Continuous_F_9 <- function(dataset1){
  Var <- c("F_10","F_11","F_12","F_13",
           "normalization_F_10","normalization_F_11","normalization_F_12","normalization_F_13",
           "normalization_F_10_exp","normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",
           "normalization_F_10_square","normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
           "F_10_root","F_11_root","F_12_root","F_13_root",
           "F_10_ln","F_11_ln","F_12_ln","F_13_ln",
           "normalization_F_10_cube","normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
           "F_10_sin","F_11_sin","F_12_sin","F_13_sin",
           "F_10_cos","F_11_cos","F_12_cos","F_13_cos",                 
           "F_10_reciprocal","F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
           "F_9","normalization_F_9","normalization_F_9_exp","normalization_F_9_square","F_9_root","F_9_ln",
           "normalization_F_9_cube","F_9_sin","F_9_cos", "F_9_reciprocal"
  )
  F_9 <- str_c("F_9",Var[1:40],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:40){
    B<-dataset1["F_9"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:41]
  colnames(A) <- F_9
  
  normalization_F_9 <- str_c("normalization_F_9",Var[1:41],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:41){
    bb<-dataset1["normalization_F_9"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:42]
  colnames(aa) <- normalization_F_9
  
  normalization_F_9_exp <- str_c("normalization_F_9_exp",Var[1:42],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:42){
    BB<-dataset1["normalization_F_9_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:43]
  colnames(AA) <- normalization_F_9_exp
  
  normalization_F_9_square<- str_c("normalization_F_9_square",Var[1:43],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:43){
    DD<-dataset1["normalization_F_9_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:44]
  colnames(D) <- normalization_F_9_square
  
  F_9_root <- str_c("F_9_root",Var[1:44],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:44){
    dd<-dataset1["F_9_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:45]
  colnames(cc) <- F_9_root
  
  F_9_ln <- str_c("F_9_ln",Var[1:45],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:45){
    ee<-dataset1["F_9_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:46]
  colnames(e) <-F_9_ln
  
  normalization_F_9_cube <- str_c("normalization_F_9_cube",Var[1:46],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:46){
    ff<-dataset1["normalization_F_9_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:47]
  colnames(f) <-normalization_F_9_cube
  
  F_9_sin <- str_c("F_9_sin",Var[1:47],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:47){
    gg<-dataset1["F_9_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:48]
  colnames(g) <-F_9_sin
  
  F_9_cos<- str_c("F_9_cos",Var[1:48],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:48){
    hh<-dataset1["F_9_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:49]
  colnames(h) <-F_9_cos
  
  F_9_reciprocal<- str_c("F_9_reciprocal",Var[1:49],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:49){
    ii<-dataset1["F_9_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:50]
  colnames(i) <-F_9_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
F9<-Continuous_F_9(data1)
dim(F9)

#F_10對其他變數做交互作用
Continuous_F_10 <- function(dataset1){
  Var <- c("F_11","F_12","F_13",
           "normalization_F_11","normalization_F_12","normalization_F_13",
           "normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",
           "normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
           "F_11_root","F_12_root","F_13_root",
           "F_11_ln","F_12_ln","F_13_ln",
           "normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
           "F_11_sin","F_12_sin","F_13_sin",
           "F_11_cos","F_12_cos","F_13_cos",                 
           "F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
           "F_10","normalization_F_10","normalization_F_10_exp","normalization_F_10_square","F_10_root","F_10_ln",
           "normalization_F_10_cube","F_10_sin","F_10_cos", "F_10_reciprocal"
  )
  F_10 <- str_c("F_10",Var[1:30],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:30){
    B<-dataset1["F_10"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:31]
  colnames(A) <- F_10
  
  normalization_F_10 <- str_c("normalization_F_10",Var[1:31],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:31){
    bb<-dataset1["normalization_F_10"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:32]
  colnames(aa) <- normalization_F_10
  
  normalization_F_10_exp <- str_c("normalization_F_10_exp",Var[1:32],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:32){
    BB<-dataset1["normalization_F_10_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:33]
  colnames(AA) <- normalization_F_10_exp
  
  normalization_F_10_square<- str_c("normalization_F_10_square",Var[1:33],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:33){
    DD<-dataset1["normalization_F_10_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:34]
  colnames(D) <- normalization_F_10_square
  
  F_10_root <- str_c("F_10_root",Var[1:34],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:34){
    dd<-dataset1["F_10_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:35]
  colnames(cc) <- F_10_root
  
  F_10_ln <- str_c("F_10_ln",Var[1:35],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:35){
    ee<-dataset1["F_10_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:36]
  colnames(e) <-F_10_ln
  
  normalization_F_10_cube <- str_c("normalization_F_10_cube",Var[1:36],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:36){
    ff<-dataset1["normalization_F_10_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:37]
  colnames(f) <-normalization_F_10_cube
  
  F_10_sin <- str_c("F_10_sin",Var[1:37],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:37){
    gg<-dataset1["F_10_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:38]
  colnames(g) <-F_10_sin
  
  F_10_cos<- str_c("F_10_cos",Var[1:38],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:38){
    hh<-dataset1["F_10_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:39]
  colnames(h) <-F_10_cos
  
  F_10_reciprocal<- str_c("F_10_reciprocal",Var[1:39],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:39){
    ii<-dataset1["F_10_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:40]
  colnames(i) <-F_10_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
F10<-Continuous_F_10(data1)
dim(F10)

#F_11對其他變數做交互作用
Continuous_F_11 <- function(dataset1){
  Var <- c("F_12","F_13",
           "normalization_F_12","normalization_F_13",
           "normalization_F_12_exp","normalization_F_13_exp",
           "normalization_F_12_square","normalization_F_13_square",
           "F_12_root","F_13_root",
           "F_12_ln","F_13_ln",
           "normalization_F_12_cube","normalization_F_13_cube",
           "F_12_sin","F_13_sin",
           "F_12_cos","F_13_cos",                 
           "F_12_reciprocal","F_13_reciprocal",
           "F_11","normalization_F_11","normalization_F_11_exp","normalization_F_11_square","F_11_root","F_11_ln",
           "normalization_F_11_cube","F_11_sin","F_11_cos", "F_11_reciprocal"
  )
  F_11 <- str_c("F_11",Var[1:20],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:20){
    B<-dataset1["F_11"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:21]
  colnames(A) <- F_11
  
  normalization_F_11 <- str_c("normalization_F_11",Var[1:21],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:21){
    bb<-dataset1["normalization_F_11"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:22]
  colnames(aa) <- normalization_F_11
  
  normalization_F_11_exp <- str_c("normalization_F_11_exp",Var[1:22],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:22){
    BB<-dataset1["normalization_F_11_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:23]
  colnames(AA) <- normalization_F_11_exp
  
  normalization_F_11_square<- str_c("normalization_F_11_square",Var[1:23],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:23){
    DD<-dataset1["normalization_F_11_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:24]
  colnames(D) <- normalization_F_11_square
  
  F_11_root <- str_c("F_11_root",Var[1:24],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:24){
    dd<-dataset1["F_11_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:25]
  colnames(cc) <- F_11_root
  
  F_11_ln <- str_c("F_11_ln",Var[1:25],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:25){
    ee<-dataset1["F_11_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:26]
  colnames(e) <-F_11_ln
  
  normalization_F_11_cube <- str_c("normalization_F_11_cube",Var[1:26],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:26){
    ff<-dataset1["normalization_F_11_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:27]
  colnames(f) <-normalization_F_11_cube
  
  F_11_sin <- str_c("F_11_sin",Var[1:27],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:27){
    gg<-dataset1["F_11_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:28]
  colnames(g) <-F_11_sin
  
  F_11_cos<- str_c("F_11_cos",Var[1:28],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:28){
    hh<-dataset1["F_11_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:29]
  colnames(h) <-F_11_cos
  
  F_11_reciprocal<- str_c("F_11_reciprocal",Var[1:29],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:29){
    ii<-dataset1["F_11_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:30]
  colnames(i) <-F_11_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
F11<-Continuous_F_11(data1)
dim(F11)
#F_12對其他變數做交互作用
Continuous_F_12 <- function(dataset1){
  Var <- c("F_13",
           "normalization_F_13",
           "normalization_F_13_exp",
           "normalization_F_13_square",
           "F_13_root",
           "F_13_ln",
           "normalization_F_13_cube",
           "F_13_sin",
           "F_13_cos",                 
           "F_13_reciprocal",
           "F_12","normalization_F_12","normalization_F_12_exp","normalization_F_12_square","F_12_root","F_12_ln",
           "normalization_F_12_cube","F_12_sin","F_12_cos", "F_12_reciprocal"
  )
  F_12 <- str_c("F_12",Var[1:10],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:10){
    B<-dataset1["F_12"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:11]
  colnames(A) <- F_12
  
  normalization_F_12 <- str_c("normalization_F_12",Var[1:11],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:11){
    bb<-dataset1["normalization_F_12"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:12]
  colnames(aa) <- normalization_F_12
  
  normalization_F_12_exp <- str_c("normalization_F_12_exp",Var[1:12],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:12){
    BB<-dataset1["normalization_F_12_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:13]
  colnames(AA) <- normalization_F_12_exp
  
  normalization_F_12_square<- str_c("normalization_F_12_square",Var[1:13],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:13){
    DD<-dataset1["normalization_F_12_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:14]
  colnames(D) <- normalization_F_12_square
  
  F_12_root <- str_c("F_12_root",Var[1:14],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:14){
    dd<-dataset1["F_12_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:15]
  colnames(cc) <- F_12_root
  
  F_12_ln <- str_c("F_12_ln",Var[1:15],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:15){
    ee<-dataset1["F_12_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:16]
  colnames(e) <-F_12_ln
  
  normalization_F_12_cube <- str_c("normalization_F_12_cube",Var[1:16],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:16){
    ff<-dataset1["normalization_F_12_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:17]
  colnames(f) <-normalization_F_12_cube
  
  F_12_sin <- str_c("F_12_sin",Var[1:17],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:17){
    gg<-dataset1["F_12_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:18]
  colnames(g) <-F_12_sin
  
  F_12_cos<- str_c("F_12_cos",Var[1:18],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:18){
    hh<-dataset1["F_12_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:19]
  colnames(h) <-F_12_cos
  
  F_12_reciprocal<- str_c("F_12_reciprocal",Var[1:19],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:19){
    ii<-dataset1["F_12_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:20]
  colnames(i) <-F_12_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
F12<-Continuous_F_12(data1)
dim(F12)
#F_13對其他變數做交互作用
Continuous_F_13 <- function(dataset1){
  Var <- c("F_13","normalization_F_13","normalization_F_13_exp","normalization_F_13_square",
           "F_13_root","F_13_ln","normalization_F_13_cube","F_13_sin","F_13_cos","F_13_reciprocal"
  )
  normalization_F_13 <- str_c("normalization_F_13",Var[1],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  bb<-dataset1["normalization_F_13"]*dataset1[(Var[1])]
  aa<-cbind(bb,aa)
  colnames(aa) <- normalization_F_13
  
  normalization_F_13_exp <- str_c("normalization_F_13_exp",Var[1:2],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:2){
    BB<-dataset1["normalization_F_13_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:3]
  colnames(AA) <- normalization_F_13_exp
  
  normalization_F_13_square<- str_c("normalization_F_13_square",Var[1:3],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:3){
    DD<-dataset1["normalization_F_13_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:4]
  colnames(D) <- normalization_F_13_square
  
  F_13_root <- str_c("F_13_root",Var[1:4],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:4){
    dd<-dataset1["F_13_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:5]
  colnames(cc) <- F_13_root
  
  F_13_ln <- str_c("F_13_ln",Var[1:5],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:5){
    ee<-dataset1["F_13_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:6]
  colnames(e) <-F_13_ln
  
  normalization_F_13_cube <- str_c("normalization_F_13_cube",Var[1:6],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:6){
    ff<-dataset1["normalization_F_13_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:7]
  colnames(f) <-normalization_F_13_cube
  
  F_13_sin <- str_c("F_13_sin",Var[1:7],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:7){
    gg<-dataset1["F_13_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:8]
  colnames(g) <-F_13_sin
  
  F_13_cos<- str_c("F_13_cos",Var[1:8],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:8){
    hh<-dataset1["F_13_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:9]
  colnames(h) <-F_13_cos
  
  F_13_reciprocal<- str_c("F_13_reciprocal",Var[1:9],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:9){
    ii<-dataset1["F_13_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:10]
  colnames(i) <-F_13_reciprocal
  
  dataset1 <- cbind(aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
F13<-Continuous_F_13(data1)
F13<-F13[,-2]
dim(F13)
Train_data<-cbind(data1,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13)#將全部的特工工程新增出來的資料跟原始訓練資料做合併
dim(Train_data)
sum(is.na(Train_data))
anyNA(Train_data)
#--------------lasso篩選變數-------------#
# install.packages("glmnet")
library(glmnet)
lasso = glmnet(x <- as.matrix(Train_data[, -14]), y = Train_data[, 14], alpha = 1,family = "gaussian")
cv.lasso <- cv.glmnet(x = as.matrix(Train_data[, -14]), y = Train_data[, 14], alpha = 1, family = "gaussian")
best.lambda <- cv.lasso$lambda.min;best.lambda
select.ind <- which(coef(cv.lasso, s = "lambda.min") != 0)
select.ind <- select.ind[-1]-1
select.varialbes <- colnames(Train_data)[select.ind]# Lasso select var
#--------------建模-------------#
Lm.Lasso <- lm(O ~ ., Train_data[, c(select.varialbes, "O")])
na.conf <- data.frame(Lm.Lasso$coefficients)
select.col<- na.omit(na.conf)
new.select.varialbes <- row.names(select.col)
new.select.varialbes <- new.select.varialbes[2:1075]
library(xgboost)
library(useful)
Formula <- O ~. -1
Train_X <- build.x(Formula, data = Train_data[,c(new.select.varialbes,"O")])
Train_y <- build.y(Formula, data = Train_data[,c(new.select.varialbes,"O")])
xgb <- xgboost(data = Train_X,
               label = Train_y,
               max.depth = 50,
               eta = 0.1,
               nthread = 4,
               nrounds = 300,
               objective = "reg:squarederror",
               eval_metric = "rmse")

#----------測試集----------#
data3 <- read.table("C:/Users/User/Desktop/2021test0831.csv",header = TRUE,sep = ",",fill=TRUE)
dim(data3)
head(data3)
str(data3)#查看資料結構(型態)
data4<-data3[,-1]
summary(data4)
sum(is.na(data4))
head(data4)

#將正規化過的跟原本的合併
data4<- cbind(data3[,-1],data4);head(data4)

#----------特徵工程----------#
##F_1~F_13正規化之後做連續變數取指數&平方&立方
library(stringr)
# 例外處理data1中F_1等於0的情況 
for (i in c(1:7222)){
  if (data4[i,"F_1"] == 0){
    data4[i,"F_1"] <- 0 + 1
  }
}
#F_1~F_13正規化
for (i in c(1:13)){
  data4[,i] <- (data4[,i]-min(data4[,i]))/(max(data4[,i])-min(data4[,i]))
  names(data4)[i]<-str_c("normalization_F_",i)
}

#將正規化過的跟原本的合併
data4<- cbind(data3[,-1],data4);head(data4)
for (i in c(1:7222)){
  if (data4[i,"F_1"] == 0){
    data4[i,"F_1"] <- 0 + 1
  }
}
Continuous_feature_engineering1 <- function(dataset){
  Var <- c("normalization_F_1","normalization_F_2","normalization_F_3","normalization_F_4",
           "normalization_F_5","normalization_F_6","normalization_F_7","normalization_F_8",
           "normalization_F_9","normalization_F_10","normalization_F_11","normalization_F_12",
           "normalization_F_13")
  Var1 <- names(data1[,1:13])
  Exp_Var <- str_c(Var,"exp",sep="_")
  Square_Var <- str_c(Var,"square",sep = "_")
  Root_Var <- str_c(Var1,"root",sep = "_") #用原本的F1-F13號開根
  ln_var <- str_c(Var1,"ln",sep = "_") #用原本的F1-F13取log以e為底
  Cube_Var <-str_c(Var,"cube",sep="_")
  sin_var <- str_c(Var1,"sin", sep = "_")
  cos_var <- str_c(Var1,"cos", sep = "_")
  reciprocal_var <- str_c(Var1, "reciprocal", sep = "_")
  exp_matrix<-exp(dataset[Var])
  square_matrix <- dataset[Var]**2
  root_matrix <- sqrt(dataset[Var1])  #用原本的F1-F13號開根
  ln_matrix <- log(dataset[Var1]) #用原本的F1-F13取log以e為底
  cube_matrix <- dataset[Var]**3
  sin_matrix <- sin(dataset[Var1])
  cos_matrix <- cos(dataset[Var1])
  reciprocal_matrix<-1/dataset[Var1]
  colnames(exp_matrix) <- Exp_Var
  colnames(square_matrix) <- Square_Var
  colnames(root_matrix) <- Root_Var
  colnames(ln_matrix) <- ln_var
  colnames(cube_matrix)<-Cube_Var
  colnames(sin_matrix) <- sin_var
  colnames(cos_matrix) <- cos_var
  colnames(reciprocal_matrix) <- reciprocal_var
  dataset <- cbind(dataset,exp_matrix, square_matrix, root_matrix, ln_matrix, cube_matrix, sin_matrix, cos_matrix, reciprocal_matrix)
  return(dataset)
}
data4<-Continuous_feature_engineering1(data4)
dim(data4)
sum(is.na(data4))
head(data4)

#F_1對其他變數做交互作用
Continuous1_F_1 <- function(dataset1){
  Var <- c( "F_2","F_3","F_4","F_5","F_6","F_7","F_8","F_9","F_10","F_11","F_12","F_13",
            "normalization_F_2","normalization_F_3","normalization_F_4","normalization_F_5",
            "normalization_F_6","normalization_F_7","normalization_F_8","normalization_F_9",
            "normalization_F_10","normalization_F_11","normalization_F_12","normalization_F_13",
            "normalization_F_2_exp","normalization_F_3_exp","normalization_F_4_exp","normalization_F_5_exp",
            "normalization_F_6_exp","normalization_F_7_exp", "normalization_F_8_exp","normalization_F_9_exp",
            "normalization_F_10_exp","normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",   
            "normalization_F_2_square","normalization_F_3_square","normalization_F_4_square","normalization_F_5_square",
            "normalization_F_6_square","normalization_F_7_square","normalization_F_8_square","normalization_F_9_square", 
            "normalization_F_10_square","normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
            "F_2_root","F_3_root","F_4_root","F_5_root",                 
            "F_6_root","F_7_root","F_8_root","F_9_root",
            "F_10_root","F_11_root","F_12_root","F_13_root",
            "F_2_ln","F_3_ln","F_4_ln","F_5_ln",
            "F_6_ln","F_7_ln","F_8_ln","F_9_ln",
            "F_10_ln","F_11_ln","F_12_ln","F_13_ln",                  
            "normalization_F_2_cube","normalization_F_3_cube","normalization_F_4_cube","normalization_F_5_cube",
            "normalization_F_6_cube","normalization_F_7_cube","normalization_F_8_cube","normalization_F_9_cube",   
            "normalization_F_10_cube","normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
            "F_2_sin","F_3_sin","F_4_sin","F_5_sin",
            "F_6_sin","F_7_sin","F_8_sin","F_9_sin",
            "F_10_sin","F_11_sin","F_12_sin","F_13_sin",
            "F_2_cos","F_3_cos","F_4_cos","F_5_cos","F_6_cos","F_7_cos",
            "F_8_cos","F_9_cos","F_10_cos","F_11_cos","F_12_cos","F_13_cos",                 
            "F_2_reciprocal","F_3_reciprocal","F_4_reciprocal","F_5_reciprocal",
            "F_6_reciprocal","F_7_reciprocal","F_8_reciprocal","F_9_reciprocal",
            "F_10_reciprocal","F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
            "F_1","normalization_F_1","normalization_F_1_exp","normalization_F_1_square","F_1_root","F_1_ln",
            "normalization_F_1_cube","F_1_sin","F_1_cos", "F_1_reciprocal"
  )
  F_1 <- str_c("F_1",Var[1:120],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:120){
    B<-dataset1["F_1"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:121]
  colnames(A) <- F_1
  
  normalization_F_1 <- str_c("normalization_F_1",Var[1:121],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:121){
    bb<-dataset1["normalization_F_1"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:122]
  colnames(aa) <- normalization_F_1
  
  normalization_F_1_exp <- str_c("normalization_F_1_exp",Var[1:122],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:122){
    BB<-dataset1["normalization_F_1_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:123]
  colnames(AA) <- normalization_F_1_exp
  
  normalization_F_1_square<- str_c("normalization_F_1_square",Var[1:123],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:123){
    DD<-dataset1["normalization_F_1_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:124]
  colnames(D) <- normalization_F_1_square
  
  F_1_root <- str_c("F_1_root",Var[1:124],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:124){
    dd<-dataset1["F_1_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:125]
  colnames(cc) <- F_1_root
  
  F_1_ln <- str_c("F_1_ln",Var[1:125],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:125){
    ee<-dataset1["F_1_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:126]
  colnames(e) <-F_1_ln
  
  normalization_F_1_cube <- str_c("normalization_F_1_cube",Var[1:126],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:126){
    ff<-dataset1["normalization_F_1_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:127]
  colnames(f) <-normalization_F_1_cube
  
  F_1_sin <- str_c("F_1_sin",Var[1:127],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:127){
    gg<-dataset1["F_1_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:128]
  colnames(g) <-F_1_sin
  
  F_1_cos<- str_c("F_1_cos",Var[1:128],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:128){
    hh<-dataset1["F_1_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:129]
  colnames(h) <-F_1_cos
  
  F_1_reciprocal<- str_c("F_1_reciprocal",Var[1:129],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:129){
    ii<-dataset1["F_1_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:130]
  colnames(i) <-F_1_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
continuous1_F1<-Continuous1_F_1(data4)
sum(is.na(continuous1_F1))
dim(continuous1_F1)

#F_2對其他變數做交互作用
Continuous1_F_2 <- function(dataset1){
  Var <- c( "F_3","F_4","F_5","F_6","F_7","F_8","F_9","F_10","F_11","F_12","F_13",
            "normalization_F_3","normalization_F_4","normalization_F_5",
            "normalization_F_6","normalization_F_7","normalization_F_8","normalization_F_9",
            "normalization_F_10","normalization_F_11","normalization_F_12","normalization_F_13",
            "normalization_F_3_exp","normalization_F_4_exp","normalization_F_5_exp",
            "normalization_F_6_exp","normalization_F_7_exp", "normalization_F_8_exp","normalization_F_9_exp",
            "normalization_F_10_exp","normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",   
            "normalization_F_3_square","normalization_F_4_square","normalization_F_5_square",
            "normalization_F_6_square","normalization_F_7_square","normalization_F_8_square","normalization_F_9_square", 
            "normalization_F_10_square","normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
            "F_3_root","F_4_root","F_5_root",                 
            "F_6_root","F_7_root","F_8_root","F_9_root",
            "F_10_root","F_11_root","F_12_root","F_13_root",
            "F_3_ln","F_4_ln","F_5_ln",
            "F_6_ln","F_7_ln","F_8_ln","F_9_ln",
            "F_10_ln","F_11_ln","F_12_ln","F_13_ln",                  
            "normalization_F_3_cube","normalization_F_4_cube","normalization_F_5_cube",
            "normalization_F_6_cube","normalization_F_7_cube","normalization_F_8_cube","normalization_F_9_cube",   
            "normalization_F_10_cube","normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
            "F_3_sin","F_4_sin","F_5_sin",
            "F_6_sin","F_7_sin","F_8_sin","F_9_sin",
            "F_10_sin","F_11_sin","F_12_sin","F_13_sin",
            "F_3_cos","F_4_cos","F_5_cos","F_6_cos","F_7_cos",
            "F_8_cos","F_9_cos","F_10_cos","F_11_cos","F_12_cos","F_13_cos",                 
            "F_3_reciprocal","F_4_reciprocal","F_5_reciprocal",
            "F_6_reciprocal","F_7_reciprocal","F_8_reciprocal","F_9_reciprocal",
            "F_10_reciprocal","F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
            "F_2","normalization_F_2","normalization_F_2_exp","normalization_F_2_square","F_2_root","F_2_ln",
            "normalization_F_2_cube","F_2_sin","F_2_cos", "F_2_reciprocal"
  )
  F_2 <- str_c("F_2",Var[1:110],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:110){
    B<-dataset1["F_2"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:111]
  colnames(A) <- F_2
  
  normalization_F_2 <- str_c("normalization_F_2",Var[1:111],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:111){
    bb<-dataset1["normalization_F_2"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:112]
  colnames(aa) <- normalization_F_2
  
  normalization_F_2_exp <- str_c("normalization_F_2_exp",Var[1:112],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:112){
    BB<-dataset1["normalization_F_2_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:113]
  colnames(AA) <- normalization_F_2_exp
  
  normalization_F_2_square<- str_c("normalization_F_2_square",Var[1:113],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:113){
    DD<-dataset1["normalization_F_2_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:114]
  colnames(D) <- normalization_F_2_square
  
  F_2_root <- str_c("F_2_root",Var[1:114],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:114){
    dd<-dataset1["F_2_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:115]
  colnames(cc) <- F_2_root
  
  F_2_ln <- str_c("F_2_ln",Var[1:115],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:115){
    ee<-dataset1["F_2_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:116]
  colnames(e) <-F_2_ln
  
  normalization_F_2_cube <- str_c("normalization_F_2_cube",Var[1:116],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:116){
    ff<-dataset1["normalization_F_2_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:117]
  colnames(f) <-normalization_F_2_cube
  
  F_2_sin <- str_c("F_2_sin",Var[1:117],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:117){
    gg<-dataset1["F_2_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:118]
  colnames(g) <-F_2_sin
  
  F_2_cos<- str_c("F_2_cos",Var[1:118],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:118){
    hh<-dataset1["F_2_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:119]
  colnames(h) <-F_2_cos
  
  F_2_reciprocal<- str_c("F_2_reciprocal",Var[1:119],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:119){
    ii<-dataset1["F_2_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:120]
  colnames(i) <-F_2_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
continuous1_F2<-Continuous1_F_2(data4)
sum(is.na(continuous1_F2))
dim(continuous1_F2)

#F_3對其他變數做交互作用
Continuous1_F_3 <- function(dataset1){
  Var <- c( "F_4","F_5","F_6","F_7","F_8","F_9","F_10","F_11","F_12","F_13",
            "normalization_F_4","normalization_F_5",
            "normalization_F_6","normalization_F_7","normalization_F_8","normalization_F_9",
            "normalization_F_10","normalization_F_11","normalization_F_12","normalization_F_13",
            "normalization_F_4_exp","normalization_F_5_exp",
            "normalization_F_6_exp","normalization_F_7_exp", "normalization_F_8_exp","normalization_F_9_exp",
            "normalization_F_10_exp","normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",   
            "normalization_F_4_square","normalization_F_5_square",
            "normalization_F_6_square","normalization_F_7_square","normalization_F_8_square","normalization_F_9_square", 
            "normalization_F_10_square","normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
            "F_4_root","F_5_root",                 
            "F_6_root","F_7_root","F_8_root","F_9_root",
            "F_10_root","F_11_root","F_12_root","F_13_root",
            "F_4_ln","F_5_ln",
            "F_6_ln","F_7_ln","F_8_ln","F_9_ln",
            "F_10_ln","F_11_ln","F_12_ln","F_13_ln",                  
            "normalization_F_4_cube","normalization_F_5_cube",
            "normalization_F_6_cube","normalization_F_7_cube","normalization_F_8_cube","normalization_F_9_cube",   
            "normalization_F_10_cube","normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
            "F_4_sin","F_5_sin",
            "F_6_sin","F_7_sin","F_8_sin","F_9_sin",
            "F_10_sin","F_11_sin","F_12_sin","F_13_sin",
            "F_4_cos","F_5_cos","F_6_cos","F_7_cos",
            "F_8_cos","F_9_cos","F_10_cos","F_11_cos","F_12_cos","F_13_cos",                 
            "F_4_reciprocal","F_5_reciprocal",
            "F_6_reciprocal","F_7_reciprocal","F_8_reciprocal","F_9_reciprocal",
            "F_10_reciprocal","F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
            "F_3","normalization_F_3","normalization_F_3_exp","normalization_F_3_square","F_3_root","F_3_ln",
            "normalization_F_3_cube","F_3_sin","F_3_cos", "F_3_reciprocal"
  )
  F_3 <- str_c("F_3",Var[1:100],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:100){
    B<-dataset1["F_3"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:101]
  colnames(A) <- F_3
  
  normalization_F_3 <- str_c("normalization_F_3",Var[1:101],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:101){
    bb<-dataset1["normalization_F_3"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:102]
  colnames(aa) <- normalization_F_3
  
  normalization_F_3_exp <- str_c("normalization_F_3_exp",Var[1:102],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:102){
    BB<-dataset1["normalization_F_3_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:103]
  colnames(AA) <- normalization_F_3_exp
  
  normalization_F_3_square<- str_c("normalization_F_3_square",Var[1:103],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:103){
    DD<-dataset1["normalization_F_3_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:104]
  colnames(D) <- normalization_F_3_square
  
  F_3_root <- str_c("F_3_root",Var[1:104],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:104){
    dd<-dataset1["F_3_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:105]
  colnames(cc) <- F_3_root
  
  F_3_ln <- str_c("F_3_ln",Var[1:105],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:105){
    ee<-dataset1["F_3_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:106]
  colnames(e) <-F_3_ln
  
  normalization_F_3_cube <- str_c("normalization_F_3_cube",Var[1:106],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:106){
    ff<-dataset1["normalization_F_3_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:107]
  colnames(f) <-normalization_F_3_cube
  
  F_3_sin <- str_c("F_3_sin",Var[1:107],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:107){
    gg<-dataset1["F_3_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:108]
  colnames(g) <-F_3_sin
  
  F_3_cos<- str_c("F_3_cos",Var[1:108],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:108){
    hh<-dataset1["F_3_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:109]
  colnames(h) <-F_3_cos
  
  F_3_reciprocal<- str_c("F_3_reciprocal",Var[1:109],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:109){
    ii<-dataset1["F_3_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:110]
  colnames(i) <-F_3_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
continuous1_F3<-Continuous1_F_3(data4)
sum(is.na(continuous1_F3))
dim(continuous1_F3)

#F_4對其他變數做交互作用
Continuous1_F_4 <- function(dataset1){
  Var <- c( "F_5","F_6","F_7","F_8","F_9","F_10","F_11","F_12","F_13",
            "normalization_F_5",
            "normalization_F_6","normalization_F_7","normalization_F_8","normalization_F_9",
            "normalization_F_10","normalization_F_11","normalization_F_12","normalization_F_13",
            "normalization_F_5_exp",
            "normalization_F_6_exp","normalization_F_7_exp", "normalization_F_8_exp","normalization_F_9_exp",
            "normalization_F_10_exp","normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",   
            "normalization_F_5_square",
            "normalization_F_6_square","normalization_F_7_square","normalization_F_8_square","normalization_F_9_square", 
            "normalization_F_10_square","normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
            "F_5_root",                 
            "F_6_root","F_7_root","F_8_root","F_9_root",
            "F_10_root","F_11_root","F_12_root","F_13_root",
            "F_5_ln",
            "F_6_ln","F_7_ln","F_8_ln","F_9_ln",
            "F_10_ln","F_11_ln","F_12_ln","F_13_ln",                  
            "normalization_F_5_cube",
            "normalization_F_6_cube","normalization_F_7_cube","normalization_F_8_cube","normalization_F_9_cube",   
            "normalization_F_10_cube","normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
            "F_5_sin",
            "F_6_sin","F_7_sin","F_8_sin","F_9_sin",
            "F_10_sin","F_11_sin","F_12_sin","F_13_sin",
            "F_5_cos","F_6_cos","F_7_cos",
            "F_8_cos","F_9_cos","F_10_cos","F_11_cos","F_12_cos","F_13_cos",                 
            "F_5_reciprocal",
            "F_6_reciprocal","F_7_reciprocal","F_8_reciprocal","F_9_reciprocal",
            "F_10_reciprocal","F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
            "F_4","normalization_F_4","normalization_F_4_exp","normalization_F_4_square","F_4_root","F_4_ln",
            "normalization_F_4_cube","F_4_sin","F_4_cos", "F_4_reciprocal"
  )
  F_4 <- str_c("F_4",Var[1:90],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:90){
    B<-dataset1["F_4"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:91]
  colnames(A) <- F_4
  
  normalization_F_4 <- str_c("normalization_F_4",Var[1:91],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:91){
    bb<-dataset1["normalization_F_4"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:92]
  colnames(aa) <- normalization_F_4
  
  normalization_F_4_exp <- str_c("normalization_F_4_exp",Var[1:92],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:92){
    BB<-dataset1["normalization_F_4_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:93]
  colnames(AA) <- normalization_F_4_exp
  
  normalization_F_4_square<- str_c("normalization_F_4_square",Var[1:93],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:93){
    DD<-dataset1["normalization_F_4_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:94]
  colnames(D) <- normalization_F_4_square
  
  F_4_root <- str_c("F_4_root",Var[1:94],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:94){
    dd<-dataset1["F_4_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:95]
  colnames(cc) <- F_4_root
  
  F_4_ln <- str_c("F_4_ln",Var[1:95],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:95){
    ee<-dataset1["F_4_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:96]
  colnames(e) <-F_4_ln
  
  normalization_F_4_cube <- str_c("normalization_F_4_cube",Var[1:96],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:96){
    ff<-dataset1["normalization_F_4_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:97]
  colnames(f) <-normalization_F_4_cube
  
  F_4_sin <- str_c("F_4_sin",Var[1:97],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:97){
    gg<-dataset1["F_4_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:98]
  colnames(g) <-F_4_sin
  
  F_4_cos<- str_c("F_4_cos",Var[1:98],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:98){
    hh<-dataset1["F_4_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:99]
  colnames(h) <-F_4_cos
  
  F_4_reciprocal<- str_c("F_4_reciprocal",Var[1:99],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:99){
    ii<-dataset1["F_4_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:100]
  colnames(i) <-F_4_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
continuous1_F4<-Continuous1_F_4(data4)
sum(is.na(continuous1_F4))
dim(continuous1_F4)

#F_5對其他變數做交互作用
Continuous1_F_5 <- function(dataset1){
  Var <- c("F_6","F_7","F_8","F_9","F_10","F_11","F_12","F_13",
           "normalization_F_6","normalization_F_7","normalization_F_8","normalization_F_9",
           "normalization_F_10","normalization_F_11","normalization_F_12","normalization_F_13",
           "normalization_F_6_exp","normalization_F_7_exp", "normalization_F_8_exp","normalization_F_9_exp",
           "normalization_F_10_exp","normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",
           "normalization_F_6_square","normalization_F_7_square","normalization_F_8_square","normalization_F_9_square", 
           "normalization_F_10_square","normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
           "F_6_root","F_7_root","F_8_root","F_9_root",
           "F_10_root","F_11_root","F_12_root","F_13_root",
           "F_6_ln","F_7_ln","F_8_ln","F_9_ln",
           "F_10_ln","F_11_ln","F_12_ln","F_13_ln",
           "normalization_F_6_cube","normalization_F_7_cube","normalization_F_8_cube","normalization_F_9_cube",   
           "normalization_F_10_cube","normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
           "F_6_sin","F_7_sin","F_8_sin","F_9_sin",
           "F_10_sin","F_11_sin","F_12_sin","F_13_sin",
           "F_6_cos","F_7_cos","F_8_cos","F_9_cos","F_10_cos","F_11_cos","F_12_cos","F_13_cos",                 
           "F_6_reciprocal","F_7_reciprocal","F_8_reciprocal","F_9_reciprocal",
           "F_10_reciprocal","F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
           "F_5","normalization_F_5","normalization_F_5_exp","normalization_F_5_square","F_5_root","F_5_ln",
           "normalization_F_5_cube","F_5_sin","F_5_cos", "F_5_reciprocal"
  )
  F_5 <- str_c("F_5",Var[1:80],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:80){
    B<-dataset1["F_5"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:81]
  colnames(A) <- F_5
  
  normalization_F_5 <- str_c("normalization_F_5",Var[1:81],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:81){
    bb<-dataset1["normalization_F_5"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:82]
  colnames(aa) <- normalization_F_5
  
  normalization_F_5_exp <- str_c("normalization_F_5_exp",Var[1:82],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:82){
    BB<-dataset1["normalization_F_5_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:83]
  colnames(AA) <- normalization_F_5_exp
  
  normalization_F_5_square<- str_c("normalization_F_5_square",Var[1:83],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:83){
    DD<-dataset1["normalization_F_5_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:84]
  colnames(D) <- normalization_F_5_square
  
  F_5_root <- str_c("F_5_root",Var[1:84],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:84){
    dd<-dataset1["F_5_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:85]
  colnames(cc) <- F_5_root
  
  F_5_ln <- str_c("F_5_ln",Var[1:85],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:85){
    ee<-dataset1["F_5_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:86]
  colnames(e) <-F_5_ln
  
  normalization_F_5_cube <- str_c("normalization_F_5_cube",Var[1:86],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:86){
    ff<-dataset1["normalization_F_5_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:87]
  colnames(f) <-normalization_F_5_cube
  
  F_5_sin <- str_c("F_5_sin",Var[1:87],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:87){
    gg<-dataset1["F_5_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:88]
  colnames(g) <-F_5_sin
  
  F_5_cos<- str_c("F_5_cos",Var[1:88],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:88){
    hh<-dataset1["F_5_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:89]
  colnames(h) <-F_5_cos
  
  F_5_reciprocal<- str_c("F_5_reciprocal",Var[1:89],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:89){
    ii<-dataset1["F_5_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:90]
  colnames(i) <-F_5_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
continuous1_F5<-Continuous1_F_5(data4)
sum(is.na(continuous1_F5))
dim(continuous1_F5)

#F_6對其他變數做交互作用
Continuous1_F_6 <- function(dataset1){
  Var <- c("F_7","F_8","F_9","F_10","F_11","F_12","F_13",
           "normalization_F_7","normalization_F_8","normalization_F_9",
           "normalization_F_10","normalization_F_11","normalization_F_12","normalization_F_13",
           "normalization_F_7_exp", "normalization_F_8_exp","normalization_F_9_exp",
           "normalization_F_10_exp","normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",
           "normalization_F_7_square","normalization_F_8_square","normalization_F_9_square", 
           "normalization_F_10_square","normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
           "F_7_root","F_8_root","F_9_root",
           "F_10_root","F_11_root","F_12_root","F_13_root",
           "F_7_ln","F_8_ln","F_9_ln",
           "F_10_ln","F_11_ln","F_12_ln","F_13_ln",
           "normalization_F_7_cube","normalization_F_8_cube","normalization_F_9_cube",   
           "normalization_F_10_cube","normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
           "F_7_sin","F_8_sin","F_9_sin",
           "F_10_sin","F_11_sin","F_12_sin","F_13_sin",
           "F_7_cos","F_8_cos","F_9_cos","F_10_cos","F_11_cos","F_12_cos","F_13_cos",                 
           "F_7_reciprocal","F_8_reciprocal","F_9_reciprocal",
           "F_10_reciprocal","F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
           "F_6","normalization_F_6","normalization_F_6_exp","normalization_F_6_square","F_6_root","F_6_ln",
           "normalization_F_6_cube","F_6_sin","F_6_cos", "F_6_reciprocal"
  )
  F_6 <- str_c("F_6",Var[1:70],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:70){
    B<-dataset1["F_6"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:71]
  colnames(A) <- F_6
  
  normalization_F_6 <- str_c("normalization_F_6",Var[1:71],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:71){
    bb<-dataset1["normalization_F_6"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:72]
  colnames(aa) <- normalization_F_6
  
  normalization_F_6_exp <- str_c("normalization_F_6_exp",Var[1:72],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:72){
    BB<-dataset1["normalization_F_6_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:73]
  colnames(AA) <- normalization_F_6_exp
  
  normalization_F_6_square<- str_c("normalization_F_6_square",Var[1:73],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:73){
    DD<-dataset1["normalization_F_6_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:74]
  colnames(D) <- normalization_F_6_square
  
  F_6_root <- str_c("F_6_root",Var[1:74],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:74){
    dd<-dataset1["F_6_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:75]
  colnames(cc) <- F_6_root
  
  F_6_ln <- str_c("F_6_ln",Var[1:75],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:75){
    ee<-dataset1["F_6_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:76]
  colnames(e) <-F_6_ln
  
  normalization_F_6_cube <- str_c("normalization_F_6_cube",Var[1:76],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:76){
    ff<-dataset1["normalization_F_6_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:77]
  colnames(f) <-normalization_F_6_cube
  
  F_6_sin <- str_c("F_6_sin",Var[1:77],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:77){
    gg<-dataset1["F_6_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:78]
  colnames(g) <-F_6_sin
  
  F_6_cos<- str_c("F_6_cos",Var[1:78],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:78){
    hh<-dataset1["F_6_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:79]
  colnames(h) <-F_6_cos
  
  F_6_reciprocal<- str_c("F_6_reciprocal",Var[1:79],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:79){
    ii<-dataset1["F_6_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:80]
  colnames(i) <-F_6_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
continuous1_F6<-Continuous1_F_6(data4)
sum(is.na(continuous1_F6))
dim(continuous1_F6)

#F_7對其他變數做交互作用
Continuous1_F_7 <- function(dataset1){
  Var <- c("F_8","F_9","F_10","F_11","F_12","F_13",
           "normalization_F_8","normalization_F_9",
           "normalization_F_10","normalization_F_11","normalization_F_12","normalization_F_13",
           "normalization_F_8_exp","normalization_F_9_exp",
           "normalization_F_10_exp","normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",
           "normalization_F_8_square","normalization_F_9_square", 
           "normalization_F_10_square","normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
           "F_8_root","F_9_root",
           "F_10_root","F_11_root","F_12_root","F_13_root",
           "F_8_ln","F_9_ln",
           "F_10_ln","F_11_ln","F_12_ln","F_13_ln",
           "normalization_F_8_cube","normalization_F_9_cube",   
           "normalization_F_10_cube","normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
           "F_8_sin","F_9_sin",
           "F_10_sin","F_11_sin","F_12_sin","F_13_sin",
           "F_8_cos","F_9_cos","F_10_cos","F_11_cos","F_12_cos","F_13_cos",                 
           "F_8_reciprocal","F_9_reciprocal",
           "F_10_reciprocal","F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
           "F_7","normalization_F_7","normalization_F_7_exp","normalization_F_7_square","F_7_root","F_7_ln",
           "normalization_F_7_cube","F_7_sin","F_7_cos", "F_7_reciprocal"
  )
  F_7 <- str_c("F_7",Var[1:60],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:60){
    B<-dataset1["F_7"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:61]
  colnames(A) <- F_7
  
  normalization_F_7 <- str_c("normalization_F_7",Var[1:61],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:61){
    bb<-dataset1["normalization_F_7"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:62]
  colnames(aa) <- normalization_F_7
  
  normalization_F_7_exp <- str_c("normalization_F_7_exp",Var[1:62],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:62){
    BB<-dataset1["normalization_F_7_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:63]
  colnames(AA) <- normalization_F_7_exp
  
  normalization_F_7_square<- str_c("normalization_F_7_square",Var[1:63],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:63){
    DD<-dataset1["normalization_F_7_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:64]
  colnames(D) <- normalization_F_7_square
  
  F_7_root <- str_c("F_7_root",Var[1:64],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:64){
    dd<-dataset1["F_7_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:65]
  colnames(cc) <- F_7_root
  
  F_7_ln <- str_c("F_7_ln",Var[1:65],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:65){
    ee<-dataset1["F_7_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:66]
  colnames(e) <-F_7_ln
  
  normalization_F_7_cube <- str_c("normalization_F_7_cube",Var[1:66],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:66){
    ff<-dataset1["normalization_F_7_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:67]
  colnames(f) <-normalization_F_7_cube
  
  F_7_sin <- str_c("F_7_sin",Var[1:67],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:67){
    gg<-dataset1["F_7_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:68]
  colnames(g) <-F_7_sin
  
  F_7_cos<- str_c("F_7_cos",Var[1:68],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:68){
    hh<-dataset1["F_7_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:69]
  colnames(h) <-F_7_cos
  
  F_7_reciprocal<- str_c("F_7_reciprocal",Var[1:69],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:69){
    ii<-dataset1["F_7_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:70]
  colnames(i) <-F_7_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
continuous1_F7<-Continuous1_F_7(data4)
sum(is.na(continuous1_F7))
dim(continuous1_F7)

#F_8對其他變數做交互作用
Continuous1_F_8 <- function(dataset1){
  Var <- c("F_9","F_10","F_11","F_12","F_13",
           "normalization_F_9",
           "normalization_F_10","normalization_F_11","normalization_F_12","normalization_F_13",
           "normalization_F_9_exp",
           "normalization_F_10_exp","normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",
           "normalization_F_9_square", 
           "normalization_F_10_square","normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
           "F_9_root",
           "F_10_root","F_11_root","F_12_root","F_13_root",
           "F_9_ln",
           "F_10_ln","F_11_ln","F_12_ln","F_13_ln",
           "normalization_F_9_cube",   
           "normalization_F_10_cube","normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
           "F_9_sin",
           "F_10_sin","F_11_sin","F_12_sin","F_13_sin",
           "F_9_cos","F_10_cos","F_11_cos","F_12_cos","F_13_cos",                 
           "F_9_reciprocal",
           "F_10_reciprocal","F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
           "F_8","normalization_F_8","normalization_F_8_exp","normalization_F_8_square","F_8_root","F_8_ln",
           "normalization_F_8_cube","F_8_sin","F_8_cos", "F_8_reciprocal"
  )
  F_8 <- str_c("F_8",Var[1:50],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:50){
    B<-dataset1["F_8"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:51]
  colnames(A) <- F_8
  
  normalization_F_8 <- str_c("normalization_F_8",Var[1:51],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:51){
    bb<-dataset1["normalization_F_8"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:52]
  colnames(aa) <- normalization_F_8
  
  normalization_F_8_exp <- str_c("normalization_F_8_exp",Var[1:52],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:52){
    BB<-dataset1["normalization_F_8_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:53]
  colnames(AA) <- normalization_F_8_exp
  
  normalization_F_8_square<- str_c("normalization_F_8_square",Var[1:53],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:53){
    DD<-dataset1["normalization_F_8_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:54]
  colnames(D) <- normalization_F_8_square
  
  F_8_root <- str_c("F_8_root",Var[1:54],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:54){
    dd<-dataset1["F_8_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:55]
  colnames(cc) <- F_8_root
  
  F_8_ln <- str_c("F_8_ln",Var[1:55],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:55){
    ee<-dataset1["F_8_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:56]
  colnames(e) <-F_8_ln
  
  normalization_F_8_cube <- str_c("normalization_F_8_cube",Var[1:56],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:56){
    ff<-dataset1["normalization_F_8_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:57]
  colnames(f) <-normalization_F_8_cube
  
  F_8_sin <- str_c("F_8_sin",Var[1:57],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:57){
    gg<-dataset1["F_8_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:58]
  colnames(g) <-F_8_sin
  
  F_8_cos<- str_c("F_8_cos",Var[1:58],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:58){
    hh<-dataset1["F_8_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:59]
  colnames(h) <-F_8_cos
  
  F_8_reciprocal<- str_c("F_8_reciprocal",Var[1:59],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:59){
    ii<-dataset1["F_8_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:60]
  colnames(i) <-F_8_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
continuous1_F8<-Continuous1_F_8(data4)
sum(is.na(continuous1_F8))
dim(continuous1_F8)

#F_9對其他變數做交互作用
Continuous1_F_9 <- function(dataset1){
  Var <- c("F_10","F_11","F_12","F_13",
           "normalization_F_10","normalization_F_11","normalization_F_12","normalization_F_13",
           "normalization_F_10_exp","normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",
           "normalization_F_10_square","normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
           "F_10_root","F_11_root","F_12_root","F_13_root",
           "F_10_ln","F_11_ln","F_12_ln","F_13_ln",
           "normalization_F_10_cube","normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
           "F_10_sin","F_11_sin","F_12_sin","F_13_sin",
           "F_10_cos","F_11_cos","F_12_cos","F_13_cos",                 
           "F_10_reciprocal","F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
           "F_9","normalization_F_9","normalization_F_9_exp","normalization_F_9_square","F_9_root","F_9_ln",
           "normalization_F_9_cube","F_9_sin","F_9_cos", "F_9_reciprocal"
  )
  F_9 <- str_c("F_9",Var[1:40],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:40){
    B<-dataset1["F_9"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:41]
  colnames(A) <- F_9
  
  normalization_F_9 <- str_c("normalization_F_9",Var[1:41],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:41){
    bb<-dataset1["normalization_F_9"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:42]
  colnames(aa) <- normalization_F_9
  
  normalization_F_9_exp <- str_c("normalization_F_9_exp",Var[1:42],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:42){
    BB<-dataset1["normalization_F_9_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:43]
  colnames(AA) <- normalization_F_9_exp
  
  normalization_F_9_square<- str_c("normalization_F_9_square",Var[1:43],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:43){
    DD<-dataset1["normalization_F_9_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:44]
  colnames(D) <- normalization_F_9_square
  
  F_9_root <- str_c("F_9_root",Var[1:44],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:44){
    dd<-dataset1["F_9_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:45]
  colnames(cc) <- F_9_root
  
  F_9_ln <- str_c("F_9_ln",Var[1:45],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:45){
    ee<-dataset1["F_9_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:46]
  colnames(e) <-F_9_ln
  
  normalization_F_9_cube <- str_c("normalization_F_9_cube",Var[1:46],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:46){
    ff<-dataset1["normalization_F_9_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:47]
  colnames(f) <-normalization_F_9_cube
  
  F_9_sin <- str_c("F_9_sin",Var[1:47],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:47){
    gg<-dataset1["F_9_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:48]
  colnames(g) <-F_9_sin
  
  F_9_cos<- str_c("F_9_cos",Var[1:48],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:48){
    hh<-dataset1["F_9_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:49]
  colnames(h) <-F_9_cos
  
  F_9_reciprocal<- str_c("F_9_reciprocal",Var[1:49],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:49){
    ii<-dataset1["F_9_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:50]
  colnames(i) <-F_9_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
continuous1_F9<-Continuous1_F_9(data4)
sum(is.na(continuous1_F9))
dim(continuous1_F9)

#F_10對其他變數做交互作用
Continuous1_F_10 <- function(dataset1){
  Var <- c("F_11","F_12","F_13",
           "normalization_F_11","normalization_F_12","normalization_F_13",
           "normalization_F_11_exp","normalization_F_12_exp","normalization_F_13_exp",
           "normalization_F_11_square","normalization_F_12_square","normalization_F_13_square",
           "F_11_root","F_12_root","F_13_root",
           "F_11_ln","F_12_ln","F_13_ln",
           "normalization_F_11_cube","normalization_F_12_cube","normalization_F_13_cube",
           "F_11_sin","F_12_sin","F_13_sin",
           "F_11_cos","F_12_cos","F_13_cos",                 
           "F_11_reciprocal","F_12_reciprocal","F_13_reciprocal",
           "F_10","normalization_F_10","normalization_F_10_exp","normalization_F_10_square","F_10_root","F_10_ln",
           "normalization_F_10_cube","F_10_sin","F_10_cos", "F_10_reciprocal"
  )
  F_10 <- str_c("F_10",Var[1:30],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:30){
    B<-dataset1["F_10"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:31]
  colnames(A) <- F_10
  
  normalization_F_10 <- str_c("normalization_F_10",Var[1:31],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:31){
    bb<-dataset1["normalization_F_10"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:32]
  colnames(aa) <- normalization_F_10
  
  normalization_F_10_exp <- str_c("normalization_F_10_exp",Var[1:32],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:32){
    BB<-dataset1["normalization_F_10_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:33]
  colnames(AA) <- normalization_F_10_exp
  
  normalization_F_10_square<- str_c("normalization_F_10_square",Var[1:33],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:33){
    DD<-dataset1["normalization_F_10_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:34]
  colnames(D) <- normalization_F_10_square
  
  F_10_root <- str_c("F_10_root",Var[1:34],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:34){
    dd<-dataset1["F_10_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:35]
  colnames(cc) <- F_10_root
  
  F_10_ln <- str_c("F_10_ln",Var[1:35],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:35){
    ee<-dataset1["F_10_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:36]
  colnames(e) <-F_10_ln
  
  normalization_F_10_cube <- str_c("normalization_F_10_cube",Var[1:36],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:36){
    ff<-dataset1["normalization_F_10_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:37]
  colnames(f) <-normalization_F_10_cube
  
  F_10_sin <- str_c("F_10_sin",Var[1:37],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:37){
    gg<-dataset1["F_10_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:38]
  colnames(g) <-F_10_sin
  
  F_10_cos<- str_c("F_10_cos",Var[1:38],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:38){
    hh<-dataset1["F_10_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:39]
  colnames(h) <-F_10_cos
  
  F_10_reciprocal<- str_c("F_10_reciprocal",Var[1:39],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:39){
    ii<-dataset1["F_10_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:40]
  colnames(i) <-F_10_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
continuous1_F10<-Continuous1_F_10(data4)
sum(is.na(continuous1_F10))
dim(continuous1_F10)

#F_11對其他變數做交互作用
Continuous1_F_11 <- function(dataset1){
  Var <- c("F_12","F_13",
           "normalization_F_12","normalization_F_13",
           "normalization_F_12_exp","normalization_F_13_exp",
           "normalization_F_12_square","normalization_F_13_square",
           "F_12_root","F_13_root",
           "F_12_ln","F_13_ln",
           "normalization_F_12_cube","normalization_F_13_cube",
           "F_12_sin","F_13_sin",
           "F_12_cos","F_13_cos",                 
           "F_12_reciprocal","F_13_reciprocal",
           "F_11","normalization_F_11","normalization_F_11_exp","normalization_F_11_square","F_11_root","F_11_ln",
           "normalization_F_11_cube","F_11_sin","F_11_cos", "F_11_reciprocal"
  )
  F_11 <- str_c("F_11",Var[1:20],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:20){
    B<-dataset1["F_11"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:21]
  colnames(A) <- F_11
  
  normalization_F_11 <- str_c("normalization_F_11",Var[1:21],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:21){
    bb<-dataset1["normalization_F_11"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:22]
  colnames(aa) <- normalization_F_11
  
  normalization_F_11_exp <- str_c("normalization_F_11_exp",Var[1:22],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:22){
    BB<-dataset1["normalization_F_11_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:23]
  colnames(AA) <- normalization_F_11_exp
  
  normalization_F_11_square<- str_c("normalization_F_11_square",Var[1:23],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:23){
    DD<-dataset1["normalization_F_11_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:24]
  colnames(D) <- normalization_F_11_square
  
  F_11_root <- str_c("F_11_root",Var[1:24],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:24){
    dd<-dataset1["F_11_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:25]
  colnames(cc) <- F_11_root
  
  F_11_ln <- str_c("F_11_ln",Var[1:25],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:25){
    ee<-dataset1["F_11_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:26]
  colnames(e) <-F_11_ln
  
  normalization_F_11_cube <- str_c("normalization_F_11_cube",Var[1:26],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:26){
    ff<-dataset1["normalization_F_11_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:27]
  colnames(f) <-normalization_F_11_cube
  
  F_11_sin <- str_c("F_11_sin",Var[1:27],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:27){
    gg<-dataset1["F_11_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:28]
  colnames(g) <-F_11_sin
  
  F_11_cos<- str_c("F_11_cos",Var[1:28],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:28){
    hh<-dataset1["F_11_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:29]
  colnames(h) <-F_11_cos
  
  F_11_reciprocal<- str_c("F_11_reciprocal",Var[1:29],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:29){
    ii<-dataset1["F_11_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:30]
  colnames(i) <-F_11_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
continuous1_F11<-Continuous1_F_11(data4)
sum(is.na(continuous1_F11))
dim(continuous1_F11)

#F_12對其他變數做交互作用
Continuous1_F_12 <- function(dataset1){
  Var <- c("F_13",
           "normalization_F_13",
           "normalization_F_13_exp",
           "normalization_F_13_square",
           "F_13_root",
           "F_13_ln",
           "normalization_F_13_cube",
           "F_13_sin",
           "F_13_cos",                 
           "F_13_reciprocal",
           "F_12","normalization_F_12","normalization_F_12_exp","normalization_F_12_square","F_12_root","F_12_ln",
           "normalization_F_12_cube","F_12_sin","F_12_cos", "F_12_reciprocal"
  )
  F_12 <- str_c("F_12",Var[1:10],sep = "_x_")
  A = matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (i in 1:10){
    B<-dataset1["F_12"]*dataset1[(Var[i])]
    A<-cbind(A,B)
  }
  A<-A[2:11]
  colnames(A) <- F_12
  
  normalization_F_12 <- str_c("normalization_F_12",Var[1:11],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (j in 1:11){
    bb<-dataset1["normalization_F_12"]*dataset1[(Var[j])]
    aa<-cbind(aa,bb)
  }
  aa<-aa[2:12]
  colnames(aa) <- normalization_F_12
  
  normalization_F_12_exp <- str_c("normalization_F_12_exp",Var[1:12],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:12){
    BB<-dataset1["normalization_F_12_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:13]
  colnames(AA) <- normalization_F_12_exp
  
  normalization_F_12_square<- str_c("normalization_F_12_square",Var[1:13],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:13){
    DD<-dataset1["normalization_F_12_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:14]
  colnames(D) <- normalization_F_12_square
  
  F_12_root <- str_c("F_12_root",Var[1:14],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:14){
    dd<-dataset1["F_12_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:15]
  colnames(cc) <- F_12_root
  
  F_12_ln <- str_c("F_12_ln",Var[1:15],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:15){
    ee<-dataset1["F_12_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:16]
  colnames(e) <-F_12_ln
  
  normalization_F_12_cube <- str_c("normalization_F_12_cube",Var[1:16],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:16){
    ff<-dataset1["normalization_F_12_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:17]
  colnames(f) <-normalization_F_12_cube
  
  F_12_sin <- str_c("F_12_sin",Var[1:17],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:17){
    gg<-dataset1["F_12_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:18]
  colnames(g) <-F_12_sin
  
  F_12_cos<- str_c("F_12_cos",Var[1:18],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:18){
    hh<-dataset1["F_12_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:19]
  colnames(h) <-F_12_cos
  
  F_12_reciprocal<- str_c("F_12_reciprocal",Var[1:19],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:19){
    ii<-dataset1["F_12_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:20]
  colnames(i) <-F_12_reciprocal
  
  dataset1 <- cbind(A,aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
continuous1_F12<-Continuous1_F_12(data4)
sum(is.na(continuous1_F12))
dim(continuous1_F12)
#F_13對其他變數做交互作用
Continuous1_F_13 <- function(dataset1){
  Var <- c("F_13","normalization_F_13","normalization_F_13_exp","normalization_F_13_square",
           "F_13_root","F_13_ln","normalization_F_13_cube","F_13_sin","F_13_cos","F_13_reciprocal"
  )
  normalization_F_13 <- str_c("normalization_F_13",Var[1],sep = "_x_")
  aa= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  bb<-dataset1["normalization_F_13"]*dataset1[(Var[1])]
  aa<-cbind(bb,aa)
  colnames(aa) <- normalization_F_13
  
  normalization_F_13_exp <- str_c("normalization_F_13_exp",Var[1:2],sep = "_x_")
  AA= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:2){
    BB<-dataset1["normalization_F_13_exp"]*dataset1[(Var[l])]
    AA<-cbind(AA,BB)
  }
  AA<-AA[2:3]
  colnames(AA) <- normalization_F_13_exp
  
  normalization_F_13_square<- str_c("normalization_F_13_square",Var[1:3],sep = "_x_")
  D= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (l in 1:3){
    DD<-dataset1["normalization_F_13_square"]*dataset1[(Var[l])]
    D<-cbind(D,DD)
  }
  D<-D[2:4]
  colnames(D) <- normalization_F_13_square
  
  F_13_root <- str_c("F_13_root",Var[1:4],sep = "_x_")
  cc= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:4){
    dd<-dataset1["F_13_root"]*dataset1[(Var[m])]
    cc<-cbind(cc,dd)
  }
  cc<-cc[2:5]
  colnames(cc) <- F_13_root
  
  F_13_ln <- str_c("F_13_ln",Var[1:5],sep = "_x_")
  e= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:5){
    ee<-dataset1["F_13_ln"]*dataset1[(Var[m])]
    e<-cbind(e,ee)
  }
  e<-e[2:6]
  colnames(e) <-F_13_ln
  
  normalization_F_13_cube <- str_c("normalization_F_13_cube",Var[1:6],sep = "_x_")
  f= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:6){
    ff<-dataset1["normalization_F_13_cube"]*dataset1[(Var[m])]
    f<-cbind(f,ff)
  }
  f<-f[2:7]
  colnames(f) <-normalization_F_13_cube
  
  F_13_sin <- str_c("F_13_sin",Var[1:7],sep = "_x_")
  g= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:7){
    gg<-dataset1["F_13_sin"]*dataset1[(Var[m])]
    g<-cbind(g,gg)
  }
  g<-g[2:8]
  colnames(g) <-F_13_sin
  
  F_13_cos<- str_c("F_13_cos",Var[1:8],sep = "_x_")
  h= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:8){
    hh<-dataset1["F_13_cos"]*dataset1[(Var[m])]
    h<-cbind(h,hh)
  }
  h<-h[2:9]
  colnames(h) <-F_13_cos
  
  F_13_reciprocal<- str_c("F_13_reciprocal",Var[1:9],sep = "_x_")
  i= matrix(0 ,ncol = 1,nrow = dim(dataset1)[1])
  for (m in 1:9){
    ii<-dataset1["F_13_reciprocal"]*dataset1[(Var[m])]
    i<-cbind(i,ii)
  }
  i<-i[2:10]
  colnames(i) <-F_13_reciprocal
  
  dataset1 <- cbind(aa,AA,D,cc,e,f,g,h,i)
  return(dataset1)
}
continuous1_F13<-Continuous1_F_13(data4)
continuous1_F13<-continuous1_F13[,-2]
sum(is.na(continuous1_F13))
dim(continuous1_F13)
Test_data<-cbind(data4,continuous1_F1,continuous1_F2,continuous1_F3,continuous1_F4,
                 continuous1_F5,continuous1_F6,continuous1_F7,continuous1_F8,continuous1_F9,
                 continuous1_F10,continuous1_F11,continuous1_F12,continuous1_F13)
dim(Test_data)#將全部的特工工程新增出來的資料跟原始測試資料做合併
sum(is.na(Test_data))
anyNA(Test_data)
#--------------預測測試資料的O值-------------#
O_pred <- predict(xgb, as.matrix(Test_data[, c(new.select.varialbes)]))
setwd("D://2021IMBD//")
getwd()
TestResult <- data.frame(O_pred)
write.table(TestResult,file="TestResult.csv",sep=",",row.names=F, na = "NA")
