---
layout: post
mathjax: true
title: BOOK RECOMMENDATION & SEARCH
---


```python
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

%matplotlib inline
plt.rcParams['figure.figsize'] = (8.0, 6.0) #setting figure size
```

Source data from: https://www.kaggle.com/jealousleopard/goodreadsbooks


```python

book_df = pd.read_csv('Entertainment/books.csv', error_bad_lines=False, parse_dates=['publication_date'])
book_df.head()
```

    b'Skipping line 3350: expected 12 fields, saw 13\nSkipping line 4704: expected 12 fields, saw 13\nSkipping line 5879: expected 12 fields, saw 13\nSkipping line 8981: expected 12 fields, saw 13\n'
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookID</th>
      <th>title</th>
      <th>authors</th>
      <th>average_rating</th>
      <th>isbn</th>
      <th>isbn13</th>
      <th>language_code</th>
      <th>num_pages</th>
      <th>ratings_count</th>
      <th>text_reviews_count</th>
      <th>publication_date</th>
      <th>publisher</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Harry Potter and the Half-Blood Prince (Harry ...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.57</td>
      <td>0439785960</td>
      <td>9780439785969</td>
      <td>eng</td>
      <td>652</td>
      <td>2095690</td>
      <td>27591</td>
      <td>9/16/2006</td>
      <td>Scholastic Inc.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Harry Potter and the Order of the Phoenix (Har...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.49</td>
      <td>0439358078</td>
      <td>9780439358071</td>
      <td>eng</td>
      <td>870</td>
      <td>2153167</td>
      <td>29221</td>
      <td>9/1/2004</td>
      <td>Scholastic Inc.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>Harry Potter and the Chamber of Secrets (Harry...</td>
      <td>J.K. Rowling</td>
      <td>4.42</td>
      <td>0439554896</td>
      <td>9780439554893</td>
      <td>eng</td>
      <td>352</td>
      <td>6333</td>
      <td>244</td>
      <td>11/1/2003</td>
      <td>Scholastic</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>Harry Potter and the Prisoner of Azkaban (Harr...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.56</td>
      <td>043965548X</td>
      <td>9780439655484</td>
      <td>eng</td>
      <td>435</td>
      <td>2339585</td>
      <td>36325</td>
      <td>5/1/2004</td>
      <td>Scholastic Inc.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>Harry Potter Boxed Set  Books 1-5 (Harry Potte...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.78</td>
      <td>0439682584</td>
      <td>9780439682589</td>
      <td>eng</td>
      <td>2690</td>
      <td>41428</td>
      <td>164</td>
      <td>9/13/2004</td>
      <td>Scholastic</td>
    </tr>
  </tbody>
</table>
</div>




```python
book_df.shape
```




    (11123, 12)




```python
book_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11123 entries, 0 to 11122
    Data columns (total 12 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   bookID              11123 non-null  int64  
     1   title               11123 non-null  object 
     2   authors             11123 non-null  object 
     3   average_rating      11123 non-null  float64
     4   isbn                11123 non-null  object 
     5   isbn13              11123 non-null  int64  
     6   language_code       11123 non-null  object 
     7     num_pages         11123 non-null  int64  
     8   ratings_count       11123 non-null  int64  
     9   text_reviews_count  11123 non-null  int64  
     10  publication_date    11123 non-null  object 
     11  publisher           11123 non-null  object 
    dtypes: float64(1), int64(5), object(6)
    memory usage: 1.0+ MB
    


```python
# check languages of books
book_df['language_code'].unique()
```




    array(['eng', 'en-US', 'fre', 'spa', 'en-GB', 'mul', 'grc', 'enm',
           'en-CA', 'ger', 'jpn', 'ara', 'nl', 'zho', 'lat', 'por', 'srp',
           'ita', 'rus', 'msa', 'glg', 'wel', 'swe', 'nor', 'tur', 'gla',
           'ale'], dtype=object)




```python
#visualize number of books in each language
lang_series = book_df['language_code'].value_counts()
plt.figure()
ax = lang_series.plot.barh(title='Number of books in each languages')
ax.set_xlabel('Language')
ax.set_ylabel('Number of books')
plt.show()
```


![png](output_7_0.png)



```python
#plot histogram of average_rating
plt.hist(book_df['average_rating'], bins=20)
plt.show()
```


![png](output_8_0.png)



```python
#get the matrix of tf-idf vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_book_titles = tfidf.fit_transform(book_df['title'])
```


```python
from sklearn.metrics.pairwise import linear_kernel      #faster than cosine_similarity
cosine = linear_kernel(tfidf_book_titles, tfidf_book_titles)
cosine
```




    array([[1.        , 0.71276342, 0.71518849, ..., 0.        , 0.        ,
            0.        ],
           [0.71276342, 1.        , 0.72940912, ..., 0.        , 0.        ,
            0.        ],
           [0.71518849, 0.72940912, 1.        , ..., 0.        , 0.        ,
            0.        ],
           ...,
           [0.        , 0.        , 0.        , ..., 1.        , 0.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 1.        ,
            0.        ],
           [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
            1.        ]])




```python
#create a datframe with title and indice
title_df = pd.DataFrame(book_df['title'])
title_df['indices'] = title_df.index
title_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>indices</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Harry Potter and the Half-Blood Prince (Harry ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Harry Potter and the Order of the Phoenix (Har...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Harry Potter and the Chamber of Secrets (Harry...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Harry Potter and the Prisoner of Azkaban (Harr...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Harry Potter Boxed Set  Books 1-5 (Harry Potte...</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
#function to get recommendations from the previous customer picked so the book_title MUST BE IN THE LIST OF EXISTING BOOK
def get_recommendations(book_title, number_of_recommendations):
    if (book_title in np.array(title_df['title'])):
        #get indice by book_title
        indice = title_df[title_df['title'] == book_title]['indices'].array[0]
        #get cosine by indice and create a dataframe, add 1 columns indices
        cosine_df = pd.DataFrame({'cos':cosine[indice]})
        cosine_df['indices'] = cosine_df.index
        #sort by cos to get to highest cosine similary
        sort_cosine_df = cosine_df.sort_values(by = 'cos', ascending=False)
       #choose list of indices have highest similary, limit by number_of_recommendation
        chosen_top_indices = sort_cosine_df[1:number_of_recommendations +1]['indices'].array
        #print out book title
        for ind in chosen_top_indices:
            print(f"{title_df[title_df['indices'] == ind]['title'].values[0]}")
    else:
        print('Sorry we dont have recommendations for your right now. Please try again later.')
    
```


```python
#try to get recommendation from the book "The Ice-Shirt (Seven Dreams #1)"
get_recommendations('Poor People', 15)
```

    The Book of Other People
    The Working Poor: Invisible in America
    All New People
    A Man of the People
    We Were Not Like Other People
    The Rainbow People
    The People of Paper
    Europe and the People Without History
    Independent People
    Pathologies of Power: Health  Human Rights and the New War on the Poor
    Banker to the Poor: Micro-Lending and the Battle Against World Poverty
    The Five People You Meet in Heaven
    The Five People You Meet in Heaven
    What Do You Care What Other People Think?
    Winning with People Workbook
    


```python
#function to search random books (book_title may not be in the existing books) and I will limit only display 10 books only
def search_book(search_book):
    #append new book and transform to vectors
    new_df = pd.DataFrame({"title":[search_book], 
                    "indices":[title_df['indices'].max()+1]})
    new_title_df = title_df.append(new_df)

    tfidf_book_titles = tfidf.fit_transform(new_title_df['title'])
    new_cosine = linear_kernel(tfidf_book_titles, tfidf_book_titles)    

    #get indice by book_title
    indice = new_title_df[new_title_df['title'] == search_book]['indices'].array[0]

     #get cosine by indice and create a dataframe, add 1 columns indices
    cosine_df = pd.DataFrame({'cos':new_cosine[indice]})
    cosine_df['indices'] = cosine_df.index

     #sort by cos to get to highest cosine similary
    sort_cosine_df = cosine_df.sort_values(by = 'cos', ascending=False)
    #choose list of indices have highest similary, limit by number_of_recommendation
    chosen_top_indices = sort_cosine_df[1:11]['indices'].array
    
    print('WE HAVE 10 RESULTS AS FOLLOWING:')
    #print out book title by indices
    for ind in chosen_top_indices:
        print(f"{new_title_df[new_title_df['indices'] == ind]['title'].values[0]}")

```


```python
#example: search random book title
search_book('Winter')
```

    WE HAVE 10 RESULTS AS FOLLOWING:
    It's Winter
    Winter's Tales
    The Winter's Tale
    Winter's Tale
    Winter (Four Seasons  #4)
    Winter on the Farm
    Brian's Winter
    Winter of Magic's Return
    The Winter of Our Discontent
    Winter Cottage
    

### Book recommendation by Title and Summary

Data from https://www.kaggle.com/ymaricar/cmu-book-summary-dataset?select=booksummaries.txt


```python
new_book_df = pd.read_csv('Entertainment/booksummaries.txt',error_bad_lines=False,delimiter="\t",header=None,\
                      names=["BookID", "Unknown", "Title","Author","Published Date","Tags","Summary"])
new_book_df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BookID</th>
      <th>Unknown</th>
      <th>Title</th>
      <th>Author</th>
      <th>Published Date</th>
      <th>Tags</th>
      <th>Summary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>620</td>
      <td>/m/0hhy</td>
      <td>Animal Farm</td>
      <td>George Orwell</td>
      <td>1945-08-17</td>
      <td>{"/m/016lj8": "Roman \u00e0 clef", "/m/06nbt":...</td>
      <td>Old Major, the old boar on the Manor Farm, ca...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>843</td>
      <td>/m/0k36</td>
      <td>A Clockwork Orange</td>
      <td>Anthony Burgess</td>
      <td>1962</td>
      <td>{"/m/06n90": "Science Fiction", "/m/0l67h": "N...</td>
      <td>Alex, a teenager living in near-future Englan...</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_book_df['title_summary'] = new_book_df['Title'] + ' ' + new_book_df['Summary']
new_book_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BookID</th>
      <th>Unknown</th>
      <th>Title</th>
      <th>Author</th>
      <th>Published Date</th>
      <th>Tags</th>
      <th>Summary</th>
      <th>title_summary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>620</td>
      <td>/m/0hhy</td>
      <td>Animal Farm</td>
      <td>George Orwell</td>
      <td>1945-08-17</td>
      <td>{"/m/016lj8": "Roman \u00e0 clef", "/m/06nbt":...</td>
      <td>Old Major, the old boar on the Manor Farm, ca...</td>
      <td>Animal Farm  Old Major, the old boar on the Ma...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>843</td>
      <td>/m/0k36</td>
      <td>A Clockwork Orange</td>
      <td>Anthony Burgess</td>
      <td>1962</td>
      <td>{"/m/06n90": "Science Fiction", "/m/0l67h": "N...</td>
      <td>Alex, a teenager living in near-future Englan...</td>
      <td>A Clockwork Orange  Alex, a teenager living in...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>986</td>
      <td>/m/0ldx</td>
      <td>The Plague</td>
      <td>Albert Camus</td>
      <td>1947</td>
      <td>{"/m/02m4t": "Existentialism", "/m/02xlf": "Fi...</td>
      <td>The text of The Plague is divided into five p...</td>
      <td>The Plague  The text of The Plague is divided ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1756</td>
      <td>/m/0sww</td>
      <td>An Enquiry Concerning Human Understanding</td>
      <td>David Hume</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>The argument of the Enquiry proceeds by a ser...</td>
      <td>An Enquiry Concerning Human Understanding  The...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2080</td>
      <td>/m/0wkt</td>
      <td>A Fire Upon the Deep</td>
      <td>Vernor Vinge</td>
      <td>NaN</td>
      <td>{"/m/03lrw": "Hard science fiction", "/m/06n90...</td>
      <td>The novel posits that space around the Milky ...</td>
      <td>A Fire Upon the Deep  The novel posits that sp...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#get the matrix of tf-idf vectors from the cleaned_summary
tfidf = TfidfVectorizer(stop_words='english')
tfidf_title_summary = tfidf.fit_transform(new_book_df['title_summary'])
cosine = linear_kernel(tfidf_title_summary, tfidf_title_summary)
cosine
```




    array([[1.        , 0.01112758, 0.01196829, ..., 0.00772858, 0.00125136,
            0.01035577],
           [0.01112758, 1.        , 0.01828531, ..., 0.0067206 , 0.00165698,
            0.01035863],
           [0.01196829, 0.01828531, 1.        , ..., 0.01026542, 0.00815554,
            0.02040999],
           ...,
           [0.00772858, 0.0067206 , 0.01026542, ..., 1.        , 0.        ,
            0.01267054],
           [0.00125136, 0.00165698, 0.00815554, ..., 0.        , 1.        ,
            0.00797861],
           [0.01035577, 0.01035863, 0.02040999, ..., 0.01267054, 0.00797861,
            1.        ]])




```python
#create a dataframe with title and indice
title_summary_df = pd.DataFrame({'title_summary':new_book_df['title_summary'], 'title':new_book_df['Title']})
title_summary_df['indices'] = title_summary_df.index
title_summary_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title_summary</th>
      <th>title</th>
      <th>indices</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Animal Farm  Old Major, the old boar on the Ma...</td>
      <td>Animal Farm</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A Clockwork Orange  Alex, a teenager living in...</td>
      <td>A Clockwork Orange</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Plague  The text of The Plague is divided ...</td>
      <td>The Plague</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>An Enquiry Concerning Human Understanding  The...</td>
      <td>An Enquiry Concerning Human Understanding</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A Fire Upon the Deep  The novel posits that sp...</td>
      <td>A Fire Upon the Deep</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
#get_recommendations by book title (from book title and summary) (10 recommendations)
def get_recommendations(book_title):
     #get indice by book_title
        indice = title_summary_df[title_summary_df['title'] == book_title]['indices'].array[0]
        cosine_df = pd.DataFrame({'cos':cosine[indice]})
        cosine_df['indices'] = cosine_df.index
        #sort by cos to get to highest cosine similary
        sort_cosine_df = cosine_df.sort_values(by = 'cos', ascending=False)
       #choose list of indices have highest similary, limit by number_of_recommendation
        chosen_top_indices = sort_cosine_df[1:11]['indices'].array
        #print out book title
        for ind in chosen_top_indices:
            print(f"{title_summary_df[title_summary_df['indices'] == ind]['title'].values[0]}")
```


```python
get_recommendations('Animal Farm')
```

    Animal Farm
    Snowball's Chance
    Fire and Sword
    Moscow 1812: Napoleon's Fatal March
    Freddy Goes to Florida
    Piggie Pie
    Jack the Hare and Mukuyu Forest
    The True Story of the Three Little Pigs
    Arctic Adventure
    The Fields of Death
    Animal World
    
