# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
# Data.csv :
```
Program Developed: B.Dhivya Shri
Register number:212221230009
import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
```
### Encoding.csv :
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4
```

# Titanic.csv :
```
import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
```
# OUPUT
# Data.csv:
![image](https://user-images.githubusercontent.com/94505585/167287271-cd3e6b3e-57c4-445c-8d6d-b3b02c83d8db.png) ![image](https://user-images.githubusercontent.com/94505585/167287278-77c82a27-7def-4b32-b012-4f5f08e39bf0.png) ![image](https://user-images.githubusercontent.com/94505585/167287281-2f0ed0d1-6cac-47ea-9ec6-b6cc0e4c83da.png)
![image](https://user-images.githubusercontent.com/94505585/167287287-cbadd4b7-e1ef-45be-8731-2e40f9740e19.png) ![image](https://user-images.githubusercontent.com/94505585/167287291-bc40a9ef-d01b-41af-98b1-cebbbe5a08eb.png) ![image](https://user-images.githubusercontent.com/94505585/167287298-4b13257b-69cd-45a6-b713-22128e5a53dd.png)
![image](https://user-images.githubusercontent.com/94505585/167287305-977a8bfe-2d68-4ad9-8976-d6af64d76ff9.png) ![image](https://user-images.githubusercontent.com/94505585/167287309-65bc0fb9-2364-44e8-bf9c-e51e7eda57d7.png)
# Encoding.csv:
![image](https://user-images.githubusercontent.com/94505585/167287326-e2cb0458-9701-4624-8b14-f8703b9ee68f.png) ![image](https://user-images.githubusercontent.com/94505585/167287335-ff8ba57f-49d3-410c-910d-fd9d17eac348.png) ![image](https://user-images.githubusercontent.com/94505585/167287348-700b6a4b-4e50-4afb-9d4b-8f88bbf0301c.png)
![image](https://user-images.githubusercontent.com/94505585/167287358-1d3f93dc-67c4-4458-a241-735604be8610.png) ![image](https://user-images.githubusercontent.com/94505585/167287362-facc98db-fe85-4314-a386-716a5c684270.png)
![image](https://user-images.githubusercontent.com/94505585/167287366-25169b01-737b-4f0f-8984-9cc5af2217b9.png) ![image](https://user-images.githubusercontent.com/94505585/167287370-b98c4a37-6397-46d4-94c8-59460927cf45.png) ![image](https://user-images.githubusercontent.com/94505585/167287377-cb6b474b-1cc3-45a1-a273-498204711e09.png)
# Titanic.csv:
![image](https://user-images.githubusercontent.com/94505585/167287392-880d5c41-b16b-48fd-8a83-0e5b8b82890c.png)
![image](https://user-images.githubusercontent.com/94505585/167287397-4c9d8097-cd74-459e-815f-ac3d8dee3636.png) ![image](https://user-images.githubusercontent.com/94505585/167287407-b6b524d7-cba0-4d36-b0bd-570951ad05e5.png) ![image](https://user-images.githubusercontent.com/94505585/167287419-3dfb7865-e938-46ba-94b5-788f6d916a79.png)
![image](https://user-images.githubusercontent.com/94505585/167287428-317b2f77-51b4-4aeb-a552-f6a9fa5fbf87.png) ![image](https://user-images.githubusercontent.com/94505585/167287450-8508b074-f112-4db6-b5c7-42254f82e850.png)
![image](https://user-images.githubusercontent.com/94505585/167287455-13b76d9b-aa73-44d3-8e09-f0ec6ef5873b.png)
![image](https://user-images.githubusercontent.com/94505585/167287466-0627fb13-c9af-436a-aae9-08154d88b357.png)
![image](https://user-images.githubusercontent.com/94505585/167287470-0a1f4544-23b0-48c2-92d6-b906ad40874d.png)
![image](https://user-images.githubusercontent.com/94505585/167287473-ef9a8031-9763-4e20-96f4-292e7b79ea81.png)
![image](https://user-images.githubusercontent.com/94505585/167287480-3b074f1a-b984-41e5-83ad-a27cb1952148.png)

# RESULT:
Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.









































