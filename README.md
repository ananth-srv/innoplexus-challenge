# innoplexus-challenge

I am sharing my approach to the Innoplexus-Challenge hosted by Analytics Vidhya https://datahack.analyticsvidhya.com/contest/innoplexus-hiring-hackathon/ on which I ended up at the 1st position in the Private leaderboard(2nd in the public leaderboard). Thank you so much for this opportunity guys to test our knowledge on machine learning.  

# Problem

This is just a 2 day challenge, the objective of the competition is to predict the list of articles which will be referenced by an article. The train dataset contains information about the article, authors, title, abstract, full text, set, published date and the list of all the articles it has as references. Test dataset as expected contains everything except the reference list.

# Approach
In this competition I mainly focussed on feature extraction. There were no direct features which we can use in our algorithm. With very less features available I thought identifying correct features will be the key in this competition. Used most of the available time in identifying features and finding out the best probability to identify the reference articles. I have just the basic knowledge in NLP and just tried to identify features using that. Xgboost has always been my first go-to model as it is fast and gets comparable results with anyother model.  

# Feature Extraction.

1. Used title, abstract, author, published date to identify the features. Since full text had data for very less number of articles I chose to ignore that field.
2. Split all the string fields to get all the words and removed the stop words and got the stem of each word. The author field was split by comma to extract the complete name of the author.
3. For each article found all the articles which belonged to the same set whose published date on or before the published date and created one row for each of them. Based on the reference list marked them if they were referenced or not.
4. Found the distance between the 

