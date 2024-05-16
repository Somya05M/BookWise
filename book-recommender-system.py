#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


books = pd.read_csv('./dataset/Books.csv')
users = pd.read_csv('./dataset/Users.csv')
ratings = pd.read_csv('./dataset/Ratings.csv')


# In[3]:


books['Image-URL-M'][1]


# In[4]:


users.head()


# In[5]:


ratings.head()


# In[6]:


print(books.shape)
print(ratings.shape)
print(users.shape)


# In[7]:


books.isnull().sum()


# In[8]:


users.isnull().sum()


# In[9]:


ratings.isnull().sum()


# In[10]:


books.duplicated().sum()


# In[11]:


ratings.duplicated().sum()


# In[12]:


users.duplicated().sum()


# In[20]:


books.head()


# ## Popularity Based Recommender System

# In[13]:


ratings_with_name = ratings.merge(books,on='ISBN')


# In[14]:


num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)
num_rating_df.head()


# In[15]:


# First, convert the 'Book-Rating' column to numeric, ignoring errors
ratings_with_name['Book-Rating'] = pd.to_numeric(ratings_with_name['Book-Rating'], errors='coerce')

# Remove rows where 'Book-Rating' couldn't be converted to numeric
ratings_with_name = ratings_with_name.dropna(subset=['Book-Rating'])

# Now, perform the groupby operation and calculate the mean
avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()

# Rename the column
avg_rating_df.rename(columns={'Book-Rating':'avg_rating'}, inplace=True)

# Display the resulting DataFrame
print(avg_rating_df)


# In[16]:


popular_df = num_rating_df.merge(avg_rating_df,on='Book-Title')
popular_df


# In[17]:


popular_df = popular_df[popular_df['num_ratings']>=250].sort_values('avg_rating',ascending=False).head(50)


# In[18]:


popular_df = popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_rating']]


# In[19]:


popular_df['Image-URL-M'][0]


# ## Collaborative Filtering Based Recommender System

# In[21]:


x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
padhe_likhe_users = x[x].index


# In[22]:


filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]


# In[23]:


y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books = y[y].index


# In[24]:


final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]


# In[25]:


pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')


# In[26]:


pt.fillna(0,inplace=True)


# In[27]:


pt


# In[28]:


from sklearn.metrics.pairwise import cosine_similarity


# In[29]:


similarity_scores = cosine_similarity(pt)


# In[30]:


similarity_scores.shape


# In[31]:


def recommend(book_name):
    # index fetch
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:5]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    
    return data


# In[32]:


recommend('1984')


# In[33]:


pt.index[545]


# In[34]:


import pickle
pickle.dump(popular_df,open('popular.pkl','wb'))


# In[35]:


books.drop_duplicates('Book-Title')


# In[36]:


pickle.dump(pt,open('pt.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_scores,open('similarity_scores.pkl','wb'))


# In[ ]:




