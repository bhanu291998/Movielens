#!/usr/bin/env python
# coding: utf-8

# # Import the Library

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # load the dataset

# In[4]:


df_user=pd.read_csv('users.dat',sep="::",names=['UserID','Gender','Age','Occupation','Zip Code'],engine ='python')


# In[5]:


df_user


# In[14]:


df_user['Gender'].value_counts().idxmax()


# In[5]:


df_user.shape


# In[6]:


df_user.isna().sum().any


# In[7]:


df_movies=pd.read_csv('movies.dat',sep="::",names=['MovieID','Title','Genres'],engine ='python')


# In[8]:


df_movies.head()


# In[9]:


df_movies.shape


# In[10]:


df_movies.info()


# In[11]:


df_movies.isna().sum()


# In[12]:


df_ratings=pd.read_csv('ratings.dat',sep="::",names=['UserID','MovieID','Title','Timestamp'],engine='python')


# In[13]:


df_ratings.shape


# In[14]:


df_ratings.isna().sum()


# In[15]:


df_ratings.head()


# # Now merge the dataset movies and ratings and then merge the resultant datset with user dataset

# In[16]:


df_MovieRatings=df_movies.merge(df_ratings,on='MovieID',how='inner')


# In[17]:


df_MovieRatings.head()


# In[18]:


df_MovieRatings.info()


# In[19]:


df_MovieRatings.shape


# In[20]:


df_MovieRatings.isna().sum()


# In[21]:


df_master=df_MovieRatings.merge(df_user,on='UserID',how='inner')


# In[22]:


df_master.info()


# In[23]:


df_master.isna().sum().any()


# In[24]:


# to csv file
df_master.to_csv('Master Data.csv')
df_master.head()


# # Explore the data using EDA

# # User Age Distribution

# In[25]:


df_master['Age'].value_counts().plot(kind='bar')
plt.xlabel('Age')
plt.title('User Age Distribution')
plt.ylabel('Users Count')
plt.show()


# In[26]:


df_master['Age'].value_counts().plot(kind='hist')
plt.xlabel('Age')
plt.title('User Age Distribution')
plt.ylabel('Users Count')
plt.show()


# as we can see from the bar graph and histogram that users around the age of 25 are the one who are more involved in giving ratings than other users this also means that they are the one who are watching more movies as compared to other users.
# also we can note that as the age is increasing , less user are involved in ratings

# # User Ratings of the movie 'Toy Story'

# In[27]:


#Extract Toy Story
toystory=df_master[df_master['Title_x'].str.contains('Toy Story')==True]


# In[28]:


toystory


# In[29]:


toystory.groupby(['Title_x','Title_y']).size()


# In[30]:


toystory.groupby(['Title_x','Title_y']).size().unstack().plot(kind='barh',legend=True)


# In[31]:


toystory.groupby(['Title_x','Title_y']).size().unstack().plot(kind='bar',legend=True)


# The above graph indicates that Toy Story 2 has been rated 5 by the most users where in the case of Toy Story 1 ratings 4 has an little edge over ratings 5. so most of the users have given 4 or 5 ratings which is positive note for the movie.

# # Top 25 movies by viewership rating# 

# In[32]:


dfTop25=df_master.groupby('Title_x').size().sort_values(ascending=False)[:25]
dfTop25


# In[33]:


plt.figure(figsize=(15,10))
dfTop25.plot(kind='barh')


# here are the top25 movies by viewership ratings .the insights that we can take from the above plot is that American Beauty is the top most movies in the viewership rating

# # Find the ratings for all the movies reviewed by for a particular user of user id = 2696

# In[34]:


user_2696 = df_master[df_master.UserID==2696]
user_2696


# # Feature Engineering:

# Use column genres:
# Find out all the unique genres (Hint: split the data in column genre making a list and then process the data to find out only the unique categories of genres)

# In[35]:


df_master['Genres']


# In[36]:


dfGenre=df_master['Genres'].str.split('|')


# In[37]:


dfGenre


# In[38]:


listgenres=set()
for genre in dfGenre:
    listgenres=listgenres.union(set(genre))


# In[39]:


listgenres


# In[40]:


len(listgenres)


# # Create a separate column for each genre category with a one-hot encoding ( 1 and 0) whether or not the movie belongs to that genre. 

# In[41]:


df_master['Genres']


# In[42]:


GeneresOnehot=df_master['Genres'].str.get_dummies('|')


# In[43]:


GeneresOnehot


# In[44]:


# concatenating the GeneresOnehot with master data


# In[45]:


df_master=pd.concat([df_master,GeneresOnehot],axis =1)


# In[46]:


df_master


# ##  Determine the features affecting the ratings of any particular movie.

# In[47]:


# convert the categorical variable Gender into nuumerical Variable
#convert gender M -0 and F-1
df_master['Gender'] = df_master['Gender'].replace("M",0)


# In[48]:


df_master['Gender']


# In[49]:


df_master['Gender']=df_master['Gender'].replace("F",1)


# In[50]:


df_master['Gender']


# In[51]:


df_master.groupby(['Gender','Title_y']).size().unstack().plot(kind='bar',legend=True)


# Male and female both have given 4 as the most ratings. and the number of male users who have given ratings is much higher than the female users

# In[52]:


#Age vs Ratings (Title_y) plot
df_master.groupby(['Age','Title_y']).size().unstack().plot(kind='bar',legend=True)


# for all the age of users ,ratings 4 is the most opted choice by the users.

# In[53]:


#Occupation vs Ratings plot
df_master.groupby(['Occupation','Title_y']).size().unstack().plot(kind='bar',legend=True)


# In[54]:


df_master.columns


# In[55]:


plt.figure(figsize=(15,10))
sns.heatmap(df_master.corr(),annot=True)


# In[56]:


# to build appropriate model to predict movie ratings
new_data = df_master


# In[57]:


new_data.columns


# In[58]:


features = new_data[['MovieID','Age','Occupation','Gender']].values


# In[59]:


features


# In[60]:


label=new_data[['Title_y']].values


# In[61]:


label


# In[62]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(features,label,test_size=0.20,random_state=42)


# In[63]:


X_train.shape


# In[64]:


X_test.shape


# In[65]:


from sklearn.linear_model import LinearRegression 
lr=LinearRegression()


# In[66]:


lr.fit(X_train,Y_train)


# In[67]:


lr.predict(X_test)


# In[68]:


Y_test


# In[69]:


y_predict=lr.predict(X_test)


# In[70]:


y_predict


# In[71]:


#error in regression
from sklearn.metrics import mean_squared_error
print('Mean Squared Error',mean_squared_error(Y_test,y_predict))


# In[72]:


from sklearn.metrics import r2_score
print('R2 score',r2_score(Y_test,y_predict))


# In[ ]:




