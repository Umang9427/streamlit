import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import seaborn as sns
import warnings
st.cache(persist=True)
st.set_option('deprecation.showPyplotGlobalUse', False)  #Do not show the deprecation warning

st.title('Dashboard for Data Visualization')
st.markdown('The dashboard will visualize the features of the review dataset')

filepath='wet_dog_food_preprocessed.csv'
dataset = pd.read_csv(filepath, index_col=0, encoding = "ISO-8859-1")


st.sidebar.title("Data Visualization for Review data")
#st.sidebar.checkbox("Show Analysis by graph", True, key=1)
options=['Top 10 brands based on reviews', 
         'Top 10 products based on reviews',
         'Heatmap of distribution of reviews over time',
         'Review count for a rating',
         'Sentence count in reviews',
         'Word count in sentences',
         'Top 20 authors based on number of reviews',
         'Top 10 brands based on ratings',
         'Average review length for a rating',
         'Top 10 products based on ratings',
         'Distribution of total word count in reviews',
         'Top 10 authors based on length of reviews']
select = st.sidebar.radio(label='Select a graph', options=options)



def plot_top_10_brands(input_data, annotate):
    """
    This function plots the top 10 brands in the dataset alongwith their review counts
    
        Input: Pandas column with brand names
        Output: A graph with top 10 brands and their review count
        
    """
    st.markdown('Top 10 brands in the dataset according to review count')
    plt.figure(figsize=(16,10))
    #Using value counts, which return counts of occurence of unique values in descending order
    brands=pd.value_counts(input_data)
    #Selecting the top 10 brands
    brands=brands[:10]
    #Accessing the indices of value counts using .index method
    brand_names=brands.index[:]
    #We cannot directly access all the values of brands, we have to run a loop
    brand_values=[]
    for i in range(len(brands)):
        brand_values.append(brands[i])
    #Using barplot with horizontal orientation and blue color palette
    #Color palette is Blues in reverse order (from high to low)
    splot=sns.barplot(x=brand_values, y=brand_names,palette="rocket",orient='h')  #orient= h means horizontal orientation
    
    #This loop adds annotation of red color to the barplot
    #If you want to annotate and the text should be out of the graph's width add some value to p.get_width(). The value is 120 here.
    for p in splot.patches:
        a=p.get_width()
        clr = 'blue'
        plt.text(annotate + a,
                 p.get_y()+0.55*p.get_height(),'{:1.0f}'.format(a), color=clr,ha='center', va='center')
    plt.xlabel("Brand Values",fontsize=18)
    plt.ylabel("Brand Names",fontsize=18)
    if select=='Top 10 brands based on reviews': 
        st.pyplot(plt)

plot_top_10_brands(dataset['brand_name'], 400)

def heatmap_of_time_reviews(input_data):
    """
    This function plots a heatmap of number of reviews over time by partitioning them into months and years with pivot table.
    
        Input: Pandas column containing date of review. Input format: dd/mm/yy
        Output: A heatmap with x-axis: years y-axis: months value: number of reviews
        
    """
    plt.figure(figsize=(18,10))
    #Converting input data into datetime and storing it in timestamp
    timestamp = pd.to_datetime(input_data)
    #Converting the datetime format to names of months and storing them in a month of type pandas series. This will be used as y axis.
    month=timestamp.dt.month_name()
    year=[]
    extra=[]
    month_name=[]
    #Splitting the date and storing year in a list. Similarly storing month timestamp to a list instead of dataframe
    #Adding an extra column which would help to count the frequency of reviews per month, by summing it we get the answer
    date=pd.DataFrame()
    j=0
    try:
        for i in input_data:
            splitted=i.split('/')
            year.append(int(splitted[2]))   #converting string to an int
            month_name.append(month.iloc[j]) #month dataframe contains the name of month for the specific row
            extra.append(int(1))   #Adding an extra column to calculate the total number of reviews in a month
            j=j+1
    except:
        for i in input_data:
            splitted=i.split('-')
            year.append(int(splitted[0]))   #converting string to an int
            month_name.append(month.iloc[j]) #month dataframe contains the name of month for the specific row
            extra.append(int(1))   #Adding an extra column to calculate the total number of reviews in a month
            j=j+1
    date['month']=month_name
    date['year']=year
    date['']=extra
    #pivot table will convert date dataframe to pivot table
    table = pd.pivot_table(date,index=['month'],columns=['year'],aggfunc=np.sum)
    #Here, we cannot sort the indices acc. to monthly order. So I have used reindex, to reorder the indices
    table=table.reindex(['January','February','March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    sns.heatmap(table,cmap="YlGnBu", annot=True, fmt="1.0f")
    plt.xlabel('Year',fontsize=20)
    plt.ylabel('month',fontsize=20)
    st.pyplot(plt)
if select == 'Heatmap of distribution of reviews over time':
    heatmap_of_time_reviews(dataset['review_date'])
   




