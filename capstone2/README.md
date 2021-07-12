# GOODREADS: AN ANALYSIS

Reading has been known to be quite important in developing the mind; to say the least, it improves brain connectivity and a person’s communication skills. Many school education systems even implement some sort of reading into their programs. However, just because it’s a part of healthy development and some school curricula does not mean that reading books is a chore and cannot be enjoyable. 

This analysis focuses on classifying some books as good or bad, so that not only can people enjoy reading but also publishing companies can look at what might be profitable.

The below is a breakdown of this README:
* [1. Data](#data)
* [2. Method](#method)
* [3. Data Wrangling](#wrangle)
* [4. EDA](eda)
* [5. Feature Engineering](#feng)
* [6. Model Summary and Conclusions](#summary)
* [7. Future Improvements](#improve)

# 1. Data <a name='data'></a>
The analysis done is based on [this dataset from Kaggle](#https://www.kaggle.com/meetnaren/goodreads-best-books). The dataset itself is based on a [Goodreads list of best books ever](#https://www.goodreads.com/list/show/1.Best_Books_Ever), voted by the users of Goodreads. The list is updated every week and was scraped in 2019.  With about 54,000 entries, this project is sufficient enough to develop a basic model. To directly view the dataset and the Goodreads list, click on the links below:

* [Kaggle Dataset](#https://www.kaggle.com/meetnaren/goodreads-best-books)
* [Goodreads List of Best Books Ever](#https://www.goodreads.com/list/show/1.Best_Books_Ever)

# 2. Method <a name='method'></a>
This is a supervised learning problem, where I sorted the dataset by its ratings and labeled the top half of the group as ‘good’ and the bottom half as ‘bad.’ Using this to train the models, I built and tuned each model, and evaluated its performance against both each other and a dummy classifier model.

# 3. Data Wrangling <a name='wrangle'></a>
[Data Wrangling Notebook](#https://github.com/scho-git/springboard/blob/main/capstone2/data_wrangling.ipynb)

* **Problem 1:** Missing values. There were quite a lot missing in the following features: edition, few in description, format, but importantly some were missing genres and pages.
  * **Solution:** Remove the ones with missing genres and pages since those are the primary features I’ll be focusing on. Remove the ones with more than 3 missing features. The rest of the missing values will be filled in as ‘missing’ so it’s not just empty.
  * **Constraints:** I would be losing the information from the books with a lot of missing values, which could be important. Additionally, since I manually decided to focus on genres and pages, perhaps the model will not efficiently learn which features are the best by itself.
  * **Results:** About 5k books were deleted, with still a good amount remaining.
* **Problem 2:** Duplicates (same books, different versions/edition, or just exact duplicates)
  * **Solution:** Keep only one version of the books, the one with the most amount of ratings.
  * **Constraints:** Loss of information since the duplicate or different edition book data is completely removed. Although there is no guarantee that the people who reviewed those duplicated entries are different people, in hindsight, it might’ve been better to average out the duplicated values such as ratings and perhaps summing the ratings/review counts.	
* **Problem 3:** Illogical data, including books with 0 pages or 0 ratings. Since some books were categorized as audiobooks, they naturally had 0 pages.
  * **Solution:** Since ratings is my main consideration of whether a book is considered ‘good’ or ‘bad,’ I kept only the books with 100 ratings. I considered 100 to be a reasonable minimum given that the median of the ratings count is 3763. The mean wasn’t considered since some book pages are disproportionately large, skewing the mean to the right.
  * **Constraints:** There could be some information loss due to my primary focus on the book pages and ratings.
* **Problem 4:** User input format and editions created all different formats. For example: if a book was a first edition, it could be written as “First”, “First Edition”, “1st”, etc.
  * **Solution:** Separate all the edition values into two categories: 1) ‘First’ by filtering all the editions that include the words “first” or “1”; and 2) ‘Other’ for all the other editions. Only the first edition was separated out because the top 4 editions were all first editions, making up a majority of the non-missing values. In order to not lose too much information from missing ones, I also added a binary feature to designate whether the edition feature was missing. 
  * **Constraints:** Because these words were filtered through a manually created RegEx filter, some cases could have slipped through and been mistakenly added to the ‘other’ category.
* **Problem 5:** Sets of books (either joint works of the author, or a book series) were also included in the dataset. Since I’d like to look at only individual books, this was problematic.
  * **Solution:** Sets of books were dropped to the best of my abilities; any format, edition, title that included the words Box(ed), Chronicles, Sets, Collections in the title were dropped after making sure they were indeed sets of books. 
  * **Constraints:** Because the titles, editions, formats had to be looked at individually or filtered through, there may have been books which actually were sets but not correctly identified, feeding the models erroneous inputs.
* **Problem 6:** Numerous genres, most of them holding little meaning since there would only be one book within this genre in the entire dataset.
  * **Solution:** Obtain the top 15 genres, and categorize the rest as ‘Other’.
  * **Constraints:** Choosing 15 to be the threshold was a manual choice, which could present to be problematic in modeling since it may not be the optimal number.
* **Problem 7:** Different languages
  * **Solution:** None. For now, I’ve left the different languages as-is in the dataset to see how the models will do, but acknowledge this could be a potential issue.
	
# 4. EDA <a name='eda'></a>
[EDA Notebook](#https://github.com/scho-git/springboard/blob/main/capstone2/eda.ipynb)

## The Ratings
The ratings distribution looks relatively normal, centered around 4.0, but it is not without outliers. 

## The Book Pages
I acknowledge that book pages is an ambiguous number since the same novel with the same word count could have different page numbers based on the publisher, the size of the book, any illustrations it might have, etc. 

The book pages are expectedly skewed to the right; half of the books had fewer than 320 pages. However, there were many outliers on the higher end, with a book going as high as 14,777 pages. Roughly half the books’ pages lie between 227 and 403 pages. According to XYZ, that would take the average reader about 6.3 to 11.2 hours to read.

## Genres
Many of the genres in the original dataset appear only once, and would not have much of a meaning/impact in the model due to its scarcity. Because of this, I focused on only the top 15 genres and grouped the rest together into a "Other" category.

Fiction reigned the dataset, making up a whopping 54.2% of the entire dataset. At almost half, fantasy was second, followed by romance. Keeping in mind that books may be of multiple genres, a majority of books (91.63%) consisted of a genre within the top 15 and a genre that’s not listed in the top 15. 

Looking at the average ratings in each genre, historical fiction is the top genre with a 4.05872 average rating. The average rating of the #1 genre, fiction, is in the middle in 7th out of the top 15 genres, at 3.97937. 

## Authors
Sometimes authors are simply great writers, and most of their works become successful. As such, it could be important to explore the authors’ information.

Some authors seem to have really great average ratings such as Jo-Anne McArthur with a 4.84 rating. Upon closer look, however, some have written only one book. As such, it can’t be determined whether they are indeed great writers or if they were lucky. Exploring authors who’ve written more than three books, the best average rated author in the dataset is Bill Watterson with 4.693571. This is expected since the author of the Calvin and Hobbes comics!

There are also authors in the dataset with numerous works. Reigning with 146 works, legendary Stephen King is at the top of that list, followed by the well-known detective author Agatha Christie.

# 5. Feature Engineering <a name='feng'></a>
[Feature Engineering Notebook](https://github.com/scho-git/springboard/blob/main/capstone2/feature_engineering.ipynb)

## Authors
Multitudes of books have multiple authors, leading to a list of authors as feature. This would cause the same problem as the original genre values, as there’s a lot of features with little value to the model due to their scarcity. As such, I was not going to use the authors’ names as a feature in my model. However, I still wanted to feed some of that information to the model, so I created the following features for the number of authors a book has: 
1. Number of authors
2. Binary feature of whether the book has more than one author (0 for no, 1 for yes)
3. Binary feature of whether the book has more than three authors (0 for no, 1 for yes)
4. Binary feature of whether the book has more than five authors (0 for no, 1 for yes)
5. Binary feature of whether the book has more than ten authors (0 for no, 1 for yes)

## Average Author Rating
In this portion, I make the big assumption that the first author listed is the primary and most important author. As such, I created four features to capture some information about the authors:
1. First author’s average book rating
2. First author’s number of works
3. Total authors’ average book ratings
4. Total authors’ numbers of books

# 6. Model Summary and Conclusions <a name='summary'></a>
[Model Notebook](https://github.com/scho-git/springboard/blob/main/capstone2/modeling.ipynb)

None of the models performed outstandingly well, and some barely did better than the baseline dummy classifier.

| Model | F-1 Score [0, 1] | Prediction Time (s) | Fit Time (s) |
| ---- | ---- | ---- | ---- |
|Baseline | 0.51, 0.50 | 0.001995 | 0.000996 |
|Logistic Regression | 0.00, 0.67 | 0.333113 | 0.002986 |
|Random Forest | 0.66, 0.64 | 56.12945 | 3.572085 |
|Ada Boost | 0.64, 0.62 | 8.274668 | 0.652254 |
|KNN | 0.60, 0.57 | 1.787197 | 23.685205 |

Note: 0 is denoted to be a bad book while 1 is categorized as a good book.

From the above summary chart, logistic regression performed the fastest in both fit and prediction time. However, due to its low F-1 score for 0 (bad books), we can see that it uniformly predicted 1 for all of its predictions, which would not be a practical model. 

The best performing model would be the random forest algorithm, with 0.66 and 0.64 F-1 scores for bad and good books, respectively. Although the fit time for the random forest is the longest one, the prediction time is much lower. Prediction time is more important given how much more often it would be run, compared to fitting the model. 

The model's performance, however, is still not where I would like it to be, and there are a few things that could be done to improve the model. 

# 7. Future Improvements <a name='improve'></a>

Since the models are still underfitting, additional features could be added to improve the model. A few examples are listed below:
1. NLP applied to the books' descriptions feature of the original dataset or add features regarding the books' languages
2. Data vision to analyze the book cover features of the original dataset
3. Better cleaning: tanh before normalization so the numbers are cleaner for the models to use. Deal with the missing features differently (lower the threshold) or add more genres.
4. Add another book dataset with more information (language, different source, book revenues)
5. Expand into a book recommendation system
