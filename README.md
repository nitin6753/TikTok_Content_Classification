# Tik Tok Video Content Classification Using Machine Learning
TikTok plans to develop a machine learning model to distinguish between claims and opinions in reported videos. Videos identified as claims will undergo further prioritization for human moderation, streamlining the review process by focusing on content more likely to violate terms of service, improving efficiency in managing vast amounts of user reports. (This is a pedagogical project)

The data set is provided by Tik Tok, [TikTokDataset.csv](TikTokDataset.csv).<br>
For more information on the dataset view the dataset dictionary file -> [DatasetDictionary.pdf](DatasetDictionary.pdf).

Initial assesment and conclusions are provided in the [PreliminaryInvestigation.ipynb](PreliminaryInvestigation.ipynb)  jupyter notebook. In this part of the project, I performed an inspection of the data provided by the Tik Tok in order to provide key data variables and anomalies to ensure the data provided is suitable for generating clear and actionable insights.<br>

Exploratory data analysis is performed in [ExploratoryDataAnalysis.ipynb](ExploratoryDataAnalysis.ipynb) file.'
>   * EDA helps to get to know the data, understand its outliers, clean its missing values, and prepare it for future modeling. 
>   * Video view and like counts are all concentrated on low end of 1,000 for opinions.Therefore, the data distribution is right-skewed, which will inform the models and model types that will be built
>   * Over 200 null values were found in 7 different columns. As a result, future modeling should consider the null values to avoid making insights that would assume complete data.<br>

Descriptive statistics and hypothesis testing is performed in [StatisticalReview.ipynb](StatisticalReview.ipynb) file. 
>   * Conducted a hypothesis test to analyze the relationship between verified_status and video_view_count.
>   * The analysis shows that there is a significant difference in number of views between TikTok videos posted by verified accounts and TikTok videos posted by unverified accounts. 
>   * As a result, these findings suggest there might be fundamental behavioral differences between these two groups of accounts: verified and unverified.

`Logistic regression` is performed in [RegressionAnalysis.ipynb](RegressionAnalysis.ipynb) file.
>   * We observed that if a user is verified, they are much more likely to post opinions. Since the end goal is to classify claims and opinions, it’s important to build a model that shows how to predict the behavior of the account type (verified) that tend to post more opinions.
>   * So, in this part of the project, the data team built a logistic regression model that predicts verified_status.
>   * Below are the model metrics:
>       * Precision: 69%
>       * Recall: 66
>       * f1-score: 66%
>   * Based on the estimated model coefficients from the logistic regression, longer videos tend to be associated with higher odds of the user being verified.

Developed a `Random Forest` and `Gradient Boosting` model to assist in the classification of videos as either claims or opinions in [ClassificationAnalysis.ipynb](ClassificationAnalysis.ipynb) file. The folder [Pickle](Pickle) contains saved machine learning models.
>   * One hot encoded the categorical variables and generated train, validation, and test datasets.
>   * Feature Engineered on dataset:
>       * Tokenization of the video transcription text variable using NLP techniques.
>   * Performed `hyperparameter tuning` and `cross validation` for both models using `GridSearchCV`.
>   * As the cost of video False negative that is video bieng misclassified as opinion is higher, so recall is chosen as best metric.
>   * Random Forest model was considered the champion model that performed well on test datasets.
>       * Performance on the test holdout data yielded near perfect scores, with only five misclassified samples out of 3,817.
>   * Subsequent analysis indicated that, as expected, the primary predictors were all related to video engagement levels, with video view count, like count, share count, and download count accounting for nearly all predictive signal in the data

The folder [Figures](Figures) contains all the plots generated during EDA performed.
