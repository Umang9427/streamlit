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
filepath='wet_dog_food_preprocessed.csv'
dataset2 = pd.read_csv(filepath, index_col=0, encoding = "ISO-8859-1")

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
plot_top_10_brands(dataset2['brand_name'], 150)



#Product names cannot be fitted in a graph because they are long. Hence we are not plotting them.
def print_top_10_products(input_data):
    """
    This function prints the top 10 products in the dataset alongwith their review counts. We are not using a graph here because
    the names of products are very long
    
        Input: Pandas column with product names
        Output: Printing top 10 products alongwith their review counts
        
    """
    plt.figure(figsize=(14,6))
    output=pd.value_counts(input_data)
    output=output.reset_index()
    output=output[:10]
    output.columns= ['product name', 'review count']
    st.table(output)
if select=='Top 10 products based on reviews':
    print_top_10_products(dataset['product_name'])
    print_top_10_products(dataset2['product_name'])



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
    heatmap_of_time_reviews(dataset2['review_date'])



def user_ratings(input_data):
    """
    This function plots the user ratings given in the dataset
    
        Input: Pandas column containing reviews provided by users
        Output: A barplot with x-axis: rating y-axis: count of rating
        
    """
    plt.figure(figsize=(8,8))
    ratings=pd.value_counts(input_data)
    ratings=ratings.sort_index(ascending=True)
    index=ratings.index[:]
    rating_counts=[]
    for i in range(len(ratings)):
        rating_counts.append(ratings[i+1])
    splot=sns.barplot(x=index,y=rating_counts,orient='v')
    for p in splot.patches:
        #Annotating the plot
        splot.annotate(format(p.get_height(), '.0f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
        #Changing width of the bars to 0.2 of original value
        current_width = p.get_width()
        diff = current_width - 0.2
        # we change the bar width
        p.set_width(0.2)
        # we recenter the bar
        p.set_x(p.get_x() + diff * .5)
        
    plt.xlabel('rating',fontsize=18)
    plt.ylabel('review_count',fontsize=18)
    st.pyplot(plt)
if select == 'Review count for a rating':
    user_ratings(dataset['rating'])
    user_ratings(dataset2['rating'])



def sentence_count(input_data):
    """
    This function plots the sentence count of reviews provided by users
    
        Input: A list containing review text 
        Output: A barplot with x-axis: Sentence count y-axis: Total reviews containing that no. of sentences
        
    """
    plt.figure(figsize=(14,10))
    sent_length=[]
    #Calculating the number of sentences in an individual string and appending it to a list
    for i in input_data:
        temp=sent_tokenize(i)
        sent_length.append(len(temp))   
    #Finding all the unique values of number of sentences and their total frequency
    sentences=pd.value_counts(sent_length)
    #Sorting the unique counts series according to the sentence count
    sentences=sentences.sort_index(ascending=True)
    index=sentences.index[:]
    #here the variable sentence count is the frequency of a specific count in the dataset
    sentence_counts=[]
    for i in range(len(sentences)):
        sentence_counts.append(sentences[index[i]])
    print(sentence_counts)
    splot=sns.barplot(x=index,y=sentence_counts,orient='v')
    for p in splot.patches:
        #annotating the plot
        splot.annotate(format(p.get_height(), '.0f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
        #Changing width of the plot to 0.6 of original
        current_width = p.get_width()
        diff = current_width - 0.6
        # we change the bar width
        p.set_width(0.6)
        # we recenter the bar
        p.set_x(p.get_x() + diff * .5)
    plt.xlabel("Sentence count",fontsize=20)
    plt.ylabel("Total count in review text",fontsize=20)
    st.pyplot(plt)
if select == 'Sentence count in reviews':
    li=[]
    for i in dataset['review_text']:
        li.append(str(i))
    sentence_count(li)
    li=[]
    for i in dataset2['review_text']:
        li.append(str(i))
    sentence_count(li)




def word_count(input_data):
    """
    This function plots the word count per sentence in reviews provided by users
    
        Input: A list containing review text
        Output: A scatter plot with x-axis: frequency of words y-axis: Number of words in a sentence
        
    """
    plt.figure(figsize=(20,15))
    sent_length=[]
    #Calculating the number of words in an individual sentence and appending it to a list
    for i in input_data:
        temp=sent_tokenize(i)
        for j in temp:
            sent_length.append(len(j.split()))
    #Finding all the unique values of number of words and their total frequency
    words=pd.value_counts(sent_length)
    #Sorting the unique counts series according to the word count
    words=words.sort_index(ascending=True)
    index=words.index[:]
    #here the variable word counts stores the frequency for a specific word length
    word_counts=[]
    for i in range(len(words)):
        word_counts.append(words[index[i]])   
        
    plt.xlabel("Sentence length",fontsize=25)
    plt.ylabel("Frequency",fontsize=25)
    plt.scatter(index, word_counts)
    st.pyplot(plt)
if select== 'Word count in sentences':
    li=[]
    for i in dataset['review_text']:
        li.append(str(i))
    word_count(li)
    li=[]
    for i in dataset2['review_text']:
        li.append(str(i))
    word_count(li)




def plot_top_20_reviewers(input_data):
    """
    This function plots the top 20 review authors in the dataset alongwith their total reviews
    
        Input: Pandas column containing name of the reviewers
        Output: A graph with top 20 reviewers and their review count
        
    """
    plt.figure(figsize=(17,8))
    #Using value counts, which return counts of occurence of unique values in descending order
    reviewers=pd.value_counts(input_data)
    #Selecting the top 20 reviewers
    reviewers=reviewers[:20]
    #Accessing the indices of value counts using .index method
    reviewer_names=reviewers.index[:]
    #We cannot directly access all the values of reviewers, we have to run a loop
    review_count=[]
    for i in range(len(reviewers)):
        review_count.append(reviewers[i])
    #Using barplot with horizontal orientation and blue color palette
    splot=sns.barplot(x=review_count, y=reviewer_names,palette="Blues_r",orient='h')
    #This loop adds annotation of red color to the barplot
    for p in splot.patches:
        a=p.get_width()
        clr = 'red'
        plt.text(4 + a, 
        p.get_y()+0.55*p.get_height(),'{:1.0f}'.format(a),color=clr,ha='center', va='center')
    plt.xlabel("Total Reviews",fontsize=18)
    plt.ylabel("Name of Reviewer",fontsize=18)
    if select == 'Top 20 authors based on number of reviews':
        st.pyplot(plt)

plot_top_20_reviewers(dataset['review_author'])
plot_top_20_reviewers(dataset2['review_author'])




def company_ratings(brand, rating, Flag):
    """
    This function plots the brands sorted on the basis of the average positive reviews they have received
    
        Input1: Pandas column with brand names
        Input2: Pandas column with ratings
        Output: A graph showing brands with their ratings
        
    """
    plt.figure(figsize=(18,8))
    #Creating a dataframe to store brand and rating together
    df=pd.DataFrame()
    df['brand']=brand
    df['rating']=rating.astype(int)
    #Grouping the columns based on brand column and doing mean for all the other columns (rating column)
    df=df.groupby('brand').mean().reset_index()
    #Sort the dataframe from highest ratings to lowest
    df.sort_values('rating',inplace=True, ascending=Flag)
    #Creating two lists for the two columns in dataframe
    brands=[]
    ratings=[]
    for i in range(min(len(df),10)):
        brands.append(str(df['brand'].iloc[i]))
        ratings.append(df['rating'].iloc[i])
    splot=sns.barplot(x=ratings,y=brands,palette="Blues_r",orient='h')
    #loop for adding annotations
    for p in splot.patches:
        a=p.get_width()
        clr = 'red'
        plt.text(0.1 + a, 
        p.get_y()+0.55*p.get_height(),'{:1.3f}'.format(a),color=clr,ha='center', va='center')
    plt.xlabel('Ratings')
    st.pyplot(plt)
if select == 'Top 10 brands based on ratings':
    #Top 10 brands based on highest ratings
    company_ratings(dataset['brand_name'], dataset['rating'], False)
    #Top 10 brands based lowest on ratings i.e. leat 10 brands
    company_ratings(dataset['brand_name'],dataset['rating'], True)
    #Top 10 brands based on highest ratings
    company_ratings(dataset2['brand_name'], dataset2['rating'], False)
    #Top 10 brands based lowest on ratings i.e. leat 10 brands
    company_ratings(dataset2['brand_name'],dataset2['rating'], True)






def rating_review_length(ratings, reviews):
    """
    This function plots the average review word length for a particular rating
    
        Input1: Pandas column with ratings
        Input2: Pandas column with review text
        Output: A graph showing rating in x axis and average word length in y axis
        
    """
    plt.figure(figsize=(14,10))
    review_length=[]
    rating_number=[]
    #Calculating the total number words in a review and adding them to a list
    for i in range(len(reviews)):
        temp=word_tokenize(reviews.iloc[i])
        review_length.append(len(temp))   
        rating_number.append(ratings.iloc[i])
    
    #Seaborn automatically averages the review length
    splot=sns.barplot(x=rating_number,y=review_length,orient='v')
    for p in splot.patches:
        #annotating the plot
        splot.annotate(format(p.get_height(), '.0f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()+1),  #Added 1 so that the annotations visualize correctly
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
        #Changing width of the plot to 0.6 of original
        current_width = p.get_width()
        diff = current_width - 0.6
        # we change the bar width
        p.set_width(0.5)
        # we recenter the bar
        p.set_x(p.get_x() + diff * .5)
    plt.xlabel("Rating",fontsize=20)
    plt.ylabel("Average review length",fontsize=20)
    st.pyplot(plt)
if select == 'Average review length for a rating':
    rating_review_length(dataset['rating'], dataset['review_text'])
    rating_review_length(dataset2['rating'], dataset2['review_text'])





def product_ratings(brand, rating, Flag):
    """
    This function returns a dataframe that has products sorted on the basis of their ratings and review count
    
        Input1: Pandas column with brand names
        Input2: Pandas column with ratings
        Input3: Flag with True or False value. False means sorting will be done descending and True means ascending
        Output: A dataframe containing sorted products according to rating
        
    """
    plt.figure(figsize=(18,8))
    #Creating a dataframe to store brand, rating and count together
    df=pd.DataFrame()
    #Below variable will help in keeping track of the count of occurences
    review_count= [1 for i in rating]
    df['brand']=brand
    df['rating']=rating.astype(int)
    df['count']=review_count
    #Grouping the columns based on brand column and doing sum for all the other columns
    df=df.groupby('brand').sum().reset_index()
    #Normalizing (averaging) the ratings
    for i in range(len(df)):
        df['rating'].iloc[i]=df['rating'].iloc[i]/df['count'].iloc[i]
    #Sort the dataframe from highest ratings to lowest
    df.sort_values(['rating', 'count'],inplace=True, ascending=Flag)
    #Creating two lists for the two columns in dataframe
    return df
if select == 'Top 10 products based on ratings':
    #Top 10 brands based on highest ratings
    pd.set_option('display.max_colwidth',-1)
    df=product_ratings(dataset['product_name'], dataset['rating'], False)
    df=df.iloc[:10]
    df=df.reset_index()
    df=df.drop(['index'], axis=1)
    st.table(df)
    df=product_ratings(dataset['product_name'], dataset['rating'], True)
    df=df.iloc[:10]
    df=df.reset_index()
    df=df.drop(['index'], axis=1)
    st.table(df)
    df=product_ratings(dataset2['product_name'], dataset2['rating'], False)
    df=df.iloc[:10]
    df=df.reset_index()
    df=df.drop(['index'], axis=1)
    st.table(df)
    df=product_ratings(dataset2['product_name'], dataset2['rating'], True)
    df=df.iloc[:10]
    df=df.reset_index()
    df=df.drop(['index'], axis=1)
    st.table(df)





def review_length(input_data):
    """
    This function shows the scatterplot of total word length of a review
    
        Input1: Pandas column with review text
        Output: A scatter plot of word length distribution
        
    """
    plt.figure(figsize=(20,15))
    review_length=[]
    #Calculating the number of words in an individual string and appending it to a list
    for i in input_data:
        temp=word_tokenize(i)
        review_length.append(len(temp))   
    #Finding all the unique values of number of words and their total frequency
    reviews=pd.value_counts(review_length)
    #Sorting the unique counts series according to the words count
    reviews=reviews.sort_index(ascending=True)
    index=reviews.index[:]
    #here the variable review counts is the frequency of a specific count in the dataset
    review_counts=[]
    for i in range(len(reviews)):
        review_counts.append(reviews[index[i]])
    plt.xlabel("word count",fontsize=20)
    plt.ylabel("Total count in review text",fontsize=20)
    plt.scatter(index, review_counts)
    st.pyplot(plt)
if select == 'Distribution of total word count in reviews':
    review_length(dataset['review_text'])
    review_length(dataset2['review_text'])





def author_review_length(author, review, Flag):
    """
    This function prints barplot of the authors and their average word length of reviews
    
        Input1: Pandas column with authors
        Input2: Pandas column with review text
        Input3: Flag with True or False value. False means sorting will be done descending and True means ascending
        Output: A graph showing brands with their ratings
        
    """
    plt.figure(figsize=(16,8))
    lengthofreview=[len(word_tokenize(i)) for i in review]
    #Below variables handles the occurences of the reviews by keeping a count
    review_count=[1 for i in review]
    #Creating a dataframe to store author, review length and count together
    df=pd.DataFrame()
    df['author']=author
    df['review_length']=lengthofreview
    df['count']=review_count
    #Grouping the columns based on author column and doing sum for all the other columns
    df=df.groupby('author').sum().reset_index()
    #Averaging the review lengths
    for i in range(len(df)):
        df['review_length'].iloc[i]=df['review_length'].iloc[i]/df['count'].iloc[i]
    #Sorting the dataframe based on review length and count
    df.sort_values(['review_length', 'count'],inplace=True, ascending=Flag)
    df2=pd.DataFrame()
    for i in range(len(df)):
        if df['count'].iloc[i]>10:  #Author must have atleast 10 reviews
            df2=df2.append(df.iloc[i])
    splot=sns.barplot(x=df2['review_length'].iloc[:10], y=df2['author'].iloc[:10], palette="Blues_r", orient='h')
    #loop for adding annotations
    for p in splot.patches:
        a=p.get_width()
        clr = 'red'
        plt.text(4 + a, 
        p.get_y()+0.55*p.get_height(),'{:1.3f}'.format(a),color=clr,ha='center', va='center')
    plt.xlabel('Average Review Length')
    st.pyplot(plt)
if select == 'Top 10 authors based on length of reviews':
    author_review_length(dataset['review_author'], dataset['review_text'], False)
    author_review_length(dataset2['review_author'], dataset2['review_text'], False)