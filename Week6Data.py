#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:47:43 2022

@author: fatemeh
"""





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:15:45 2021

@author: bf101616
"""





import numpy as np

from sklearn.model_selection import StratifiedKFold
import numpy
# fix random seed for reproducibility




import math


# import xgboost as xgb
# from xgboost.sklearn import XGBClassifier
# from xgboost import XGBClassifier


from sklearn.impute import SimpleImputer
from pprint import pprint


#import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.cross_validation import train_test_split 
import numpy as np 

#train_data = pd.read_csv("train.csv")
#test_data = pd.read_csv("test.csv")

#y_train = train_data["Survived"]
#train_data.drop(labels="Survived", axis=1, inplace=True)

#full_data = train_data.append(test_data)
#drop_columns = ["Name", "Age", "SibSp", "Ticket", "Cabin", "Parch", "Embarked"]
#full_data.drop(labels=drop_columns, axis=1, inplace=True)
#full_data = pd.get_dummies(full_data, columns=["Sex"])
#full_data.fillna(value=0.0, inplace=True)
#X_train = full_data.values[0:891]
#X_test = full_data.values[891:]


#X=data.loc[0:880,'Group':'Vmax'].values 802
#X=X.replace('',np.nan)
#import matplotlib.pyplot as plt
#import seaborn as sns
# roc curve and auc score
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
#import Tkinter as tk





from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
#from sklearn.grid_search import GridSearchCV

import numpy as np 
#import matplotlib.pyplot as plt 



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection  import train_test_split  #mohemen in shekli bashe






import pandas as pd
#data=pd.read_csv("prebp40.csv",delimiter=";" ,decimal="."),sep='\t'
data=pd.read_csv("prebp40002.csv"  )


#from sklearn.model_selection import train_test_split




# data['Group']=data['Group'].replace('Nice',0)
# data['Group']=data['Group'].replace('Helsinki',1)
# data['Group']=data['Group'].replace('FINLAND',2)

 
#data['Athlete']=data['Athlete'].replace('','np.nan')


data['Weekly_number']=data['Weekly_number'].replace('',0)
data['Weekly_trainingtime']=data['Weekly_trainingtime'].replace('',0)
data['Weekly_trainingintensity']=data['Weekly_trainingintensity'].replace('',0)
data['Weekly_competitiontime']=data['Weekly_competitiontime'].replace('',0)
data['Weekly_competitionintensity']=data['Weekly_competitionintensity'].replace('',0)
data['Weekly_othersporttrainingtime']=data['Weekly_othersporttrainingtime'].replace('',0)
data['Weekly_othersportrainingintensity']=data['Weekly_othersportrainingintensity'].replace('',0)




data['Weekly_AIPPnumber']=data['Weekly_AIPPnumber'].replace('','np.nan')
data['Weekly_1atigue']=data['Weekly_1atigue'].replace('','np.nan')

data['Weekly_sleep']=data['Weekly_sleep'].replace('','np.nan')

data['Weekly_no-athleticinjury']=data['Weekly_no-athleticinjury'].replace('','np.nan')

data['AWeekly_Illnesses (score 0 to 3)']=data['Weekly_Illnesses (score 0 to 3)'].replace('','np.nan')

#data['Athlete']=data['Athlete'].replace('','np.nan')




#Weekly_AIPPnumber	Weekly_1atigue	Weekly_sleep	Weekly_no-athleticinjury
#	Weekly_Illnesses (score 0 to 3)

#Weekly_number	Weekly_trainingtime 	Weekly_trainingintensity	Weekly_competitiontime	Weekly_competitionintensity 	Weekly_othersporttrainingtime
#	Weekly_othersportrainingintensity
data['Baseline_sex']=data['Baseline_sex'].replace('F','1')
#data['Baseline_sex']=data['Baseline_sex'].replace('M','0')



data['Baseline_discipline']=data['Baseline_discipline'].replace('Lancers',	0)
data['Baseline_discipline']=data['Baseline_discipline'].replace('Course sur route',	1)
data['Baseline_discipline']=data['Baseline_discipline'].replace('Sauts',	2)
data['Baseline_discipline']=data['Baseline_discipline'].replace('Demi-fond et Fond (piste)',	3)
data['Baseline_discipline']=data['Baseline_discipline'].replace('Demi-1ond et 1ond (piste)',	3)


data['Baseline_discipline']=data['Baseline_discipline'].replace('Marche Athlétique',	4)
data['Baseline_discipline']=data['Baseline_discipline'].replace('Trail',	5)
data['Baseline_discipline']=data['Baseline_discipline'].replace('Haies',	6)
data['Baseline_discipline']=data['Baseline_discipline'].replace('Epreuves combinées',	7)
data['Baseline_discipline']=data['Baseline_discipline'].replace('Sprints',	8)

data['Baseline_previousseasoninjury']=data['Baseline_previousseasoninjury'].replace('NON, Aucune blessure ni problème physique la saison dernière',	0)
data['Baseline_previousseasoninjury']=data['Baseline_previousseasoninjury'].replace('NON, Aucune blessure ni la saison dernière',	0)


data['Baseline_previousseasoninjury']=data['Baseline_previousseasoninjury'].replace('OUI, mais participation complète à l entrainement ou en compétition, malgré une blessure/problème physique',	1)
data['Baseline_previousseasoninjury']=data['Baseline_previousseasoninjury'].replace('OUI, mais participation réduite à l entrainement ou en compétition, à ...',	1)

data['Baseline_previousseasoninjury']=data['Baseline_previousseasoninjury'].replace('OUI, mais participation réduite à l entrainement ou en compétition, à cause d une blessure/problème physique',	1)

data['Baseline_previousseasoninjury']=data['Baseline_previousseasoninjury'].replace('OUI, mais aucune participation possible à l entrainement ou en compéti ...',	1)

data['Weekly_trainingintensity']=data['Weekly_trainingintensity'].replace('Facile',	0)
data['Weekly_competitiontime']=data['Weekly_competitiontime'].replace('Facile',	0)
data['Weekly_competitionintensity']=data['Weekly_competitionintensity'].replace('Facile',	0)



data['Weekly_trainingintensity']=data['Weekly_trainingintensity'].replace('1acile',	0)
data['Weekly_competitiontime']=data['Weekly_competitiontime'].replace('1acile',	0)
data['Weekly_competitionintensity']=data['Weekly_competitionintensity'].replace('1acile',	0)

	
data['Weekly_trainingintensity']=data['Weekly_trainingintensity'].replace('Moyen',	1)
data['Weekly_competitiontime']=data['Weekly_competitiontime'].replace('Moyen',	1)
data['Weekly_competitionintensity']=data['Weekly_competitionintensity'].replace('Moyen',	1)
	


data['Weekly_trainingintensity']=data['Weekly_trainingintensity'].replace('Moyen',	1)
data['Weekly_competitiontime']=data['Weekly_competitiontime'].replace('Moyen',	1)
data['Weekly_competitionintensity']=data['Weekly_competitionintensity'].replace('Moyen',	1)
	

data['Weekly_trainingintensity']=data['Weekly_trainingintensity'].replace('Difficile',	2)
data['Weekly_competitiontime']=data['Weekly_competitiontime'].replace('Difficile',	2)
data['Weekly_competitionintensity']=data['Weekly_competitionintensity'].replace('Difficile',	2)



data['Weekly_trainingintensity']=data['Weekly_trainingintensity'].replace('Di11icile',	2)
data['Weekly_competitiontime']=data['Weekly_competitiontime'].replace('Di11icile',	2)
data['Weekly_competitionintensity']=data['Weekly_competitionintensity'].replace('Di11icile',	2)



##

data['Weekly_othersporttrainingtime']=data['Weekly_othersporttrainingtime'].replace('Facile',	0)
data['Weekly_othersportrainingintensity']=data['Weekly_othersportrainingintensity'].replace('Facile',	0)
#data['Weekly_othersporttrainingtime']=data['Weekly_competitionintensity'].replace('Facile',	0)



data['Weekly_othersporttrainingtime']=data['Weekly_othersporttrainingtime'].replace('1acile',	0)
data['Weekly_othersportrainingintensity']=data['Weekly_othersportrainingintensity'].replace('1acile',	0)
#data['Weekly_competitionintensity']=data['Weekly_competitionintensity'].replace('1acile',	0)

	
data['Weekly_othersporttrainingtime']=data['Weekly_othersporttrainingtime'].replace('Moyen',	1)
data['Weekly_othersportrainingintensity']=data['Weekly_othersportrainingintensity'].replace('Moyen',	1)
#data['Weekly_competitionintensity']=data['Weekly_competitionintensity'].replace('Moyen',	1)
	


data['Weekly_othersporttrainingtime']=data['Weekly_othersporttrainingtime'].replace('Moyen',	1)
data['Weekly_othersportrainingintensity']=data['Weekly_othersportrainingintensity'].replace('Moyen',	1)
#data['Weekly_competitionintensity']=data['Weekly_competitionintensity'].replace('Moyen',	1)
	

data['Weekly_othersporttrainingtime']=data['Weekly_othersporttrainingtime'].replace('Difficile',	2)
data['Weekly_othersportrainingintensity']=data['Weekly_othersportrainingintensity'].replace('Difficile',	2)
#data['Weekly_competitionintensity']=data['Weekly_competitionintensity'].replace('Difficile',	2)



data['Weekly_othersporttrainingtime']=data['Weekly_othersporttrainingtime'].replace('Di11icile',	2)
data['Weekly_othersportrainingintensity']=data['Weekly_othersportrainingintensity'].replace('Di11icile',	2)
#data['Weekly_competitionintensity']=data['Weekly_competitionintensity'].replace('Di11icile',	2)
#	Weekly_othersporttrainingtime	Weekly_othersportrainingintensity






data['Baseline_sex']=data['Baseline_sex'].replace('M',0)


#select_color = data.loc[data['Athlete'] == 1  & (data['Shape'] == 'Rectangle')]
#select_color = data.loc[data['Index'] == 1  ]
#c = select_color.loc[delect_color['Athlete'] == 1  & (data['Shape'] == 'Rectangle')]

#select_color = select_color['Index' == 1  ]



# #select_color = data.loc[data['Athlete'] == 1  & (data['Shape'] == 'Outcome1_Weekly_athleticinjury)]
# select_color = data.loc[data['Athlete'] == 1  ]

# select_color=np.array(select_color)
# #print (select_color)
# #X1=select_color[0]
# S1=select_color[0]

# X1=S1[1:23]
# Y1=S1[0]


X1=np.zeros(32)
Y1=np.zeros(1)
D=np.zeros(10)

for i in range(63):
#select_color = data.loc[data['Athlete'] == 1  & (data['Shape'] == 'Outcome1_Weekly_athleticinjury)]
    select_color = data.loc[data['Athlete'] == 1  ]
    
    select_color=np.array(select_color)
    #print (select_color)
    #X1=select_color[0]
    S1=np.transpose(select_color[0])  
    S11= np.hstack((S1[1:23], D))

    X1= np.vstack((X1, S11))
    #X1= np.hstack((X1, D))

   # X1[i]=np.array(S1[1:23])
    print(i)
    Y1= np.vstack((Y1, S1[0]))
    #Y1[i]=S1[0]

#
# np.all(select_color,axis=0)
# #Y=data.loc[3:781,'Outcome1_Weekly_athleticinjury'].values
#Y=data['Outcome1_Weekly_athleticinjury'].values
#Y1=np.transpose(Y1)  

#X=data.loc[0:2339,'Athlete':'Weekly_Illnesses (score 0 to 3)'].values

#Y=data.loc['Outcome1_Weekly_athleticinjury'].values

#Outcome1_Weekly_athleticinjury'

    
# for i in range(Y1.shape[0]):
#    for j in range(X1.shape[1]):
#    # for j in range(500):

    
#         #print(i)
#         #print(j)
#         print(X[i,j])
#         #print(math.floor(X[i,j]))
#         #print(float()int(X[i,j]))
# #        X[i,j]=float(X[i,j])
  
    













# print(Y)
# #imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# print(X)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
#imp = IterativeImputer(max_iter=10, random_state=0)
from sklearn.impute import KNNImputer
imp = KNNImputer(n_neighbors=2, weights="uniform")
imp.fit(X1)
SimpleImputer()
# X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X1))
 #imp.fit(Y)
 #SimpleImputer()
# X = [[np.nan, 2], [6, np.nan], [7, 6]]







X1=imp.transform(X1)
 #Y=imp.transform(Y)
imp.fit(Y1.reshape(-1, 1))
Y1 = np.array(imp.transform(Y1.reshape(-1, 1)))
 #print(Y)
Y1=np.transpose(Y1)
#Y=np.reshape(Y, 40)
 #Y=np.array([Y[0],Y[1])
#print(Y)
 #Y=np.array([Y[0],Y[1],Y[2],Y[3],Y[4],Y[5],Y[6],Y[7],Y[8],Y[9],Y[10],Y[11],Y[12],Y[13],Y[14],Y[15],Y[16],Y[17],Y[18],Y[19],Y[20],Y[21],Y[22],Y[23],Y[24],Y[25],Y[26],Y[27],Y[28],Y##[29],Y[30],Y[31],Y[32],Y[33],Y[34],Y[35],Y[36]])
print(Y1)
Y1=np.transpose(Y1)



for i in range(Y1.shape[0]):
#    Y1[i]=math.floor(Y1[i])

#print(Y1)

######














 X2=np.zeros(32)
Y2=np.zeros(1)
D=np.zeros(10)

for i in range(63):
#select_color = data.loc[data['Athlete'] == 1  & (data['Shape'] == 'Outcome1_Weekly_athleticinjury)]
    select_color = data.loc[data['Athlete'] == 2  ]
    
    select_color=np.array(select_color)
    #print (select_color)
    #X2=select_color[0]
    S1=np.transpose(select_color[0])  
    S11= np.hstack((S1[1:23], D))
    X2= np.vstack((X2, S11))
    #X2= np.hstack((X2, D))

   # X2[i]=np.array(S1[1:23])
    print(i)
    Y2= np.vstack((Y2, S1[0]))

    #Y2[i]=S1[0]



    













# print(Y)
# #imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# print(X)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
#imp = IterativeImputer(max_iter=10, random_state=0)
from sklearn.impute import KNNImputer
imp = KNNImputer(n_neighbors=2, weights="uniform")
imp.fit(X2)
SimpleImputer()
# X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X2))
 #imp.fit(Y)
 #SimpleImputer()
# X = [[np.nan, 2], [6, np.nan], [7, 6]]







X2=imp.transform(X2)
 #Y=imp.transform(Y)
imp.fit(Y2.reshape(-1, 1))
Y2 = np.array(imp.transform(Y2.reshape(-1, 1)))
 #print(Y)
Y2=np.transpose(Y2)
#Y=np.reshape(Y, 40)
 #Y=np.array([Y[0],Y[1])
#print(Y)
 #Y=np.array([Y[0],Y[1],Y[2],Y[3],Y[4],Y[5],Y[6],Y[7],Y[8],Y[9],Y[10],Y[11],Y[12],Y[13],Y[14],Y[15],Y[16],Y[17],Y[18],Y[19],Y[20],Y[21],Y[22],Y[23],Y[24],Y[25],Y[26],Y[27],Y[28],Y##[29],Y[30],Y[31],Y[32],Y[33],Y[34],Y[35],Y[36]])
print(Y2)
Y2=np.transpose(Y2)




for i in range(Y2.shape[0]):
#    Y2[i]=math.floor(Y2[i])

#print(Y1)
###############

#################@


 X3=np.zeros(32)
Y3=np.zeros(1)
D=np.zeros(10)

for i in range(63):
#select_color = data.loc[data['Athlete'] == 1  & (data['Shape'] == 'Outcome1_Weekly_athleticinjury)]
    select_color = data.loc[data['Athlete'] == 3  ]
    
    select_color=np.array(select_color)
    #print (select_color)
    #X3=select_color[0]
    S1=np.transpose(select_color[0])  
    S11= np.hstack((S1[1:23], D))
    X3= np.vstack((X3, S11))
    #X3= np.hstack((X3, D))

   # X3[i]=np.array(S1[1:23])
    print(i)
    Y3= np.vstack((Y3, S1[0]))

    #Y3[i]=S1[0]





# print(Y)
# #imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# print(X)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
#imp = IterativeImputer(max_iter=10, random_state=0)
from sklearn.impute import KNNImputer
imp = KNNImputer(n_neighbors=2, weights="uniform")
imp.fit(X3)
SimpleImputer()
# X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X3))
 #imp.fit(Y)
 #SimpleImputer()
# X = [[np.nan, 2], [6, np.nan], [7, 6]]







X3=imp.transform(X3)
 #Y=imp.transform(Y)
imp.fit(Y3.reshape(-1, 1))
Y3 = np.array(imp.transform(Y3.reshape(-1, 1)))
 #print(Y)
Y3=np.transpose(Y3)
#Y=np.reshape(Y, 40)
 #Y=np.array([Y[0],Y[1])
#print(Y)
 #Y=np.array([Y[0],Y[1],Y[2],Y[3],Y[4],Y[5],Y[6],Y[7],Y[8],Y[9],Y[10],Y[11],Y[12],Y[13],Y[14],Y[15],Y[16],Y[17],Y[18],Y[19],Y[20],Y[21],Y[22],Y[23],Y[24],Y[25],Y[26],Y[27],Y[28],Y##[29],Y[30],Y[31],Y[32],Y[33],Y[34],Y[35],Y[36]])
print(Y3)
Y3=np.transpose(Y3)



for i in range(Y3.shape[0]):
    Y3[i]=math.floor(Y3[i])

#print(Y3)

#####4






X4=np.zeros(32)
Y4=np.zeros(1)
D=np.zeros(10)

for i in range(63):
#select_color = data.loc[data['Athlete'] == 1  & (data['Shape'] == 'Outcome1_Weekly_athleticinjury)]
    select_color = data.loc[data['Athlete'] == 3  ]
    
    select_color=np.array(select_color)
    #print (select_color)
    #X3=select_color[0]
    S1=np.transpose(select_color[0])  
    S11= np.hstack((S1[1:23], D))
    X4= np.vstack((X4, S11))
    #X3= np.hstack((X3, D))

   # X3[i]=np.array(S1[1:23])
    print(i)
    Y4= np.vstack((Y4, S1[0]))

    #Y3[i]=S1[0]





from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
#imp = IterativeImputer(max_iter=10, random_state=0)
from sklearn.impute import KNNImputer
imp = KNNImputer(n_neighbors=2, weights="uniform")
imp.fit(X4)
SimpleImputer()
# X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X4))
 #imp.fit(Y)
 #SimpleImputer()
# X = [[np.nan, 2], [6, np.nan], [7, 6]]







X4=imp.transform(X4)
 #Y=imp.transform(Y)
imp.fit(Y4.reshape(-1, 1))
Y4 = np.array(imp.transform(Y4.reshape(-1, 1)))
 #print(Y)
Y4=np.transpose(Y4)
#Y=np.reshape(Y, 40)
 #Y=np.array([Y[0],Y[1])
#print(Y)
 #Y=np.array([Y[0],Y[1],Y[2],Y[3],Y[4],Y[5],Y[6],Y[7],Y[8],Y[9],Y[10],Y[11],Y[12],Y[13],Y[14],Y[15],Y[16],Y[17],Y[18],Y[19],Y[20],Y[21],Y[22],Y[23],Y[24],Y[25],Y[26],Y[27],Y[28],Y##[29],Y[30],Y[31],Y[32],Y[33],Y[34],Y[35],Y[36]])
print(Y4)
Y4=np.transpose(Y4)



for i in range(Y4.shape[0]):
   Y4[i]=math.floor(Y4[i])

