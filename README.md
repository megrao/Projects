# Projects - Couple of interesting notebooks and reports

## Project 1 - Face recognition
According to Yan, Kriegman, and Ahuja, face detection can be categorized into feature-based, appearance-based, knowledge-based and template matching. As the name suggests, feature-based relies on structural features of the face, whereas knowledge-based relies on the pre-existing knowledge and rules pertaining to the face constitution. Template matching correlates the input images with pre-defined, parametrized templates whereas appearance-based banks on a training set to generate face models.

The last method (appearance-based) has superior performance as it involves statistical analysis and machine learning. Another application of this method would be face feature extraction.

Sub-methods of Appearance-based methodology
Following are the sub-methods and we are planning to employ Eigen-based method and SVM, among others, for our analysis.

Eigenface-based method:- has been around since 1991 and uses Principal Component Analysis (PCA) to efficiently represent faces. PCA happens to be the linear dimensionality reduction using approximated Singular Value Decomposition of the data and keeping only the most significant singular vectors to project the data to a lower dimensional space.

Distribution-based method:- The algorithms like PCA and Fisher’s Discriminant can be used to define the subspace representing facial patterns. There is a trained classifier, which correctly identifies instances of the target pattern class from the background image patterns.

Neural-Networks:- Many detection problems like object detection, face detection, emotion detection, and face recognition, etc. have been faced successfully by Neural Networks.

Support Vector Machine (SVM):- Support Vector Machines are linear classifiers that maximise the margin between the decision hyperplane and the examples in the training set. Osuna et al. first applied this classifier to face detection.

Sparse Network of Winnows:- They defined a sparse network of two linear units or target nodes; one represents face patterns and other for the non-face patterns. It is less time consuming and efficient.

Naive Bayes Classifiers:- They computed the probability of a face to be present in the picture by counting the frequency of occurrence of a series of the pattern over the training images. The classifier captured the joint statistics of local appearance and position of the faces.

Hidden Markov Model:- The states of the model would be the facial features, which usually described as strips of pixels. HMM’s commonly used along with other methods to build detection algorithms.

Information Theoretical Approach:- Markov Random Fields (MRF) can use for face pattern and correlated features. The Markov process maximises the discrimination between classes using Kullback-Leibler divergence. Therefore this method can be used in Face Detection.

Inductive Learning:- This approach has been used to detect faces. Algorithms like Quinlan’s C4.5 or Mitchell’s FIND-S used for this purpose.

Source for the above data: http://faculty.ucmerced.edu/mhyang/facedetection.html, https://towardsdatascience.com/face-detection-for-beginners-e58e8f21aad9, http://lijiancheng0614.github.io/scikit-learn/modules/generated/sklearn.decomposition.RandomizedPCA.html

The code was adapted from the below source, published in 2013 by Olivier Grisel. I express my sincere gratitude to the contributor.

https://github.com/ogrisel/notebooks/blob/master/Labeled%20Faces%20in%20the%20Wild%20recognition.ipynb http://nbviewer.jupyter.org/github/ogrisel/notebooks/blob/master/Labeled%20Faces%20in%20the%20Wild%20recognition.ipynb

## Project 2 - Avocado Price Prediction

### Abstract
Renowned statistician and management guru late W. Edwards Deming rightly said - "If you do not know how to ask the right question, you discover nothing." Therefore, it is essential to define the problem and carefully interrogate the data with the right questions, before proceeding to perform the necessary pre-processing of the sample as per the research design. Following are the details of the project undertaken.
Topic: Analysis and Prediction of Avocado prices (specifically Hass Avocados)
Industry: Retail 
Data: The dataset comprises of attributes such as the Average sales price per unit, Date of observation, Total volume, Total bags and volume breakup for each bag size and product code of Hass avocados. Here is the link to the dataset homepage - http://www.hassavocadoboard.com/retail/volume-and-price-data and https://www.kaggle.com/neuromusic/avocado-prices.
### Problem Statement: Find the following insights from the data: 
	Effect of avocado type (conventional or organic) on pricing. Furthermore, explore if we can reverse engineer and build a model to predict the type of avocado in front of us, based on price.
	Understand the price fluctuation across different regions. Where was the maximum price?
	Study the price distribution and fluctuation in the past few years. Find the most expensive year.
	Predict and forecast of future prices. Will it increase or decrease?

### Acknowledgement
First and foremost, many thanks to Hass Avocado board for the dataset and author Brain Godsey for his book ‘Think Like a Data Scientist: Tackle the data science process step-by-step’. I express my gratitude towards my professor Dr. Glen Mules for the useful links, encouragement and guidance. Furthermore, I thank all the publishers of useful links and codes online.

### Brief Overview 

	First things first. What is a Hass Avocado?
A delicious variant of avocado, also known as Bilse avocado, with dark green-colored, bumpy skin. 
	Why this topic?
Predictive analytics was routinely performed during my undergraduate study in Bioinformatics engineering. At work, we used to generate reports to enable users to analyze and predict fuel and commodity prices. Hence the interest.
	Why is subject matter important?
I consider Avocado price as a unit of study. Price analysis and prediction can be done with any product or commodities irrespective of field. The questions we ask may slightly differ, but the basic process remains the same. Hence, it has countless applications that makes it important. On a lighter note, Avocados are tasty and nutritious, and worthy of study.
	What are you hoping to achieve?  
Reiterating the problem statement, the goal is to unearth valuable insights from the data. Below are some specific goals. 
	Effect of avocado type (conventional or organic) on pricing. Furthermore, explore if we can reverse engineer and build a model to predict the type of avocado in front of us, based on price.
	Understand the price fluctuation across different regions. Where was the maximum price?
	Study the price distribution and fluctuation in the past few years. Find the most expensive year?
	Predict and forecast of future prices. Will it increase or decrease?
Upon completion of this study, I hope to learn how to approach and handle data in the initial step of data preparation.

Dataset Studied

	Source URL: http://www.hassavocadoboard.com/retail/volume-and-price-data and https://www.kaggle.com/neuromusic/avocado-prices
	The dataset comprises of retail volume and price data of conventional and organic Hass avocados in the United States. We are excluding global data from this study. Nevertheless, the Hass Avocado Board website carries several insightful datasets for future research.

### Description & Structure of the Data

	Layout

o	Quoting directly from the data owners, below is an excerpt from the website of Hass Avocado Board describing the data on their website.
The table represents weekly 2018 retail scan data for National retail volume (units) and price. Retail scan data comes directly from retailers’ cash registers based on actual retail sales of Hass avocados. Starting in 2013, the table below reflects an expanded, multi-outlet retail data set. Multi-outlet reporting includes an aggregation of the following channels: grocery, mass, club, drug, dollar and military. The Average Price (of avocados) in the table reflects a per unit (per avocado) cost, even when multiple units (avocados) are sold in bags. The Product Lookup codes (PLU) in the table are only for Hass avocados. Other varieties of avocados (e.g. greenskins) are not included in this table.

o	Domains of attribute values: The columns in the initial and final table table are as follows.
	Date - The date of the observation
	Average Price - the average price of a single avocado
	Total Volume - Total number of avocados sold
	4046 - Total number of avocados with PLU 4046 sold
	4225 - Total number of avocados with PLU 4225 sold
	4770 - Total number of avocados with PLU 4770 sold
	Total Bags – Total number of avocado bags sold 
	Small Bags - Total number of small avocado bags sold
	Large Bags - Total number of large avocado bags sold
	XLarge Bags - Total number of extra-large avocado bags sold

Consolidated table from Kaggle.com has the following additional entries.
	Unnamed field: probably meant with Index number
	Year: Year of sale
	Region: Region of sale
	Type: Specifies if it is Conventional or Organic Avocado

### Notes on Exploration of the dataset

	There are 14 columns and 18,249 entries in the consolidated ‘Avocado.csv’ file from Kaggle dataset. 
	Dataset has the records from January 2015 to March 2018.
	The initial dataset from Hass Avocado board is in excel format with fewer columns. From visual examination, the additional columns in the consolidated sheet are Type (specifying if conventional or organic), Region and Year. Hence, we can append data in the future by filtering out latest data, adding the aforementioned appending to existing consolidated dataset ‘avocado.csv’.
	A quick study using Tableau software revealed that all regions are well-represented and there is sufficient yearly data.  

#### Explore and document the data and data structures manually 

o	The product lookup codes (PLU) codes were studied. Below are the details.
	4046 – Hass Avocado - small
	4225 – Hass Avocado - large
	4770 – Hass Avocado - Extra Large
o	Index column in the consolidated sheet from Kaggle can be eliminated as there are many inconsistent entries. It is also unnamed in the csv file.
o	As per the python pre-processing (using Jupyter notebook), there are no null value rows in the dataset. 
o	Certain columns do have zero (for example No. of Extra-large bags) but it looks genuine and correct.
o	Certain columns representing number of avocados is in float format which is unlikely to be true as number of items must ideally have integer format.

### Questions to Ask of the Dataset

o	Why was the data collected?
Data was collected for research purpose by Hass Avocado Board. They routinely perform various projections, market analysis and forecast.

o	How was the data collected?
As per their (Hass Avocado board or HAB) website, data is collected from IRI/FreshLook Marketing Multi-Outlet (MULO) retail scans. The reporting reflects retail sales scans across the following channels: grocery, mass, club, drug, dollar and military. HAB's calculation is based in part on data reported by Information Resources, Inc. through its Freshlook Service.

Do we have the required license, credentials or permission to access the data? (Castle, 2018) Yes, it is an open dataset and I verified using the website’s Terms of use section.

o	Is the data reliable and accurate? Is there any missing data?
Again, as per the HAB website, the information was believed to be reliable at the time supplied by IRI even though it is not legally guaranteed. Without limiting the generality of the foregoing, specific data points may vary from other information sources. Generally, it is deemed to be reliable and fairly accurate. There is no missing data as per initial analysis.

o	Is the data up to date? 
Yes, it is updated every four weeks. For the purpose of study, we are using entries from January 2015 to March 2018.

o	Do we have tools and network connections to implement this experiment?
Yes, we do.

o	What is the data format of the datasets? Is there any disparity that requires standardization of data (to convert to a single workable format)?
The consolidated sheet from Kaggle Open dataset is in csv (column separated) format and data from HAB website is in excel format. It is very easy converting between these two formats. Hence no issues.

o	What is the size of the dataset? Do we need to create a subset for granular analysis? (Castle, 2018) Furthermore, Is there data for future study? 
Dataset has 14 columns and 18,249 entries. We have ample data, thanks to the HAB website. There is sufficient data for current as well as future studies. The website itself provides smaller subsets of data that can be easily downloaded and used for any granular analysis. If needed, we can also work on creating a smaller subset using Python, R or SQL.

o	Is there a need to join tables or create a summary table? Can it be done manually or using computational tools?
The HAB website has smaller subsets of data for each year, region and type. Fortunately, we can use the consolidated dataset named ‘avocado.csv’ in Kaggle open dataset. There is no need to manually modify the table. 


o	Do we need more data to perform the current study? Is there a need to merge and consolidate data? Or to download data from outside the organization? (Castle, 2018)
At the expense of sounding redundant, yes, there is sufficient data for current as well as future studies. There are various types of historical data in HAB’s website. However, we are relying on Kaggle’s consolidated dataset as they have painstakingly merged regional information of over 330 entries each, totaling to over 18,000 entries, and generously uploaded in their repository for academic purposes. To perform certain tasks during the research execution phase, we may need to use computational tools (aka python in this case) to segment and segregate the data.

o	Do we have too much data to process? Will it be time-consuming? (Godsey, 2017)
No, we have entries 18,249 entries that can be easily processed without much delay.

o	Is the data accessed or modified in the production environment? Are there business users to be notified? Is there a downtime?
No to all three. We are working with an open dataset downloadable in csv and excel format. Hence, there is no need to notify users or plan a downtime. We are not toying with sensitive data or environment.

o	Any assumption?
For the purpose of this study, we are assuming a single supplier for the avocado stock, in order to avoid undue complications.

o	Are there inconsistent entries or redundant that require cleaning of the data? Can it be cleaned in the current environment with pre-existing tools and software? Is it possible to clean it manually? (Castle, 2018)
Yes, there are couple of minor tweaks, discovered and documented in the data exploration step, to be performed. Even though manual cleaning sounds easy, python is more efficient in handling this task. Hence, we are planning to employ Jupyter notebook for the same.

o	Can the data at hand address our goals?
Yes, we have the necessary data to answer the problems statements / goals and provide business insights.

o	Is the proposed methodology compatible with the data format? Is it efficient?
Yes, data format is compatible, and the methods proposed are efficient.

o	What happens if a test fails? Is there a spot-check? If it shows an error, how do we proceed? (Godsey, 2017)
We are not planning to rely on a single method. There are multiple ways and we can verify solutions from time-to-time using alternate and manual methods. If solution is wrong, we will substitute the main method with the alternate method.

o	What inferences can we make? What truth is revealed or predicted? (questions suggested by Professor Glen Mules)
Whether organic is more expensive compared to conventional Hass Avocados (we know the truth but let’s allow data to verify!). Furthermore, based on the price, experiment if we can build a model to predict the type of avocado in front of us. The average price distribution and yearly fluctuation across regions educates us about the market trends, and the forecast tells us what to expect in the future (whether the prices will increase or decrease in 2019 and 2020). Please refer Jupyter notebook for the complete results. Few screenshots below.

### Insights

Couple of interesting insights from the execution part. 
	As per the forecast, the prices are most likely to drop in the year 2019 and 2020.
	San Francisco had the highest Avocado price in the year 2016. The type was organic Avocados. It comes as no surprise that Organic are pricier but sells in lower volumes.
	Nevertheless, Hartford Springfield had the highest Average Price (followed by San Francisco and New York).
	West region is the highest consumer of Hass Avocados. State of California is the frontrunner among the states in US.
### References

	Castle, E. (2018, August 29). 6-questions-to-ask-when-preparing-data-analysis [Web log post]. Retrieved November 11,2108 from /https://www.sisense.com/blog/
	Godsey, B. (2017). Think Like a Data Scientist: Tackle the Data Science Process Step-By-Step. Manning Publications.
	OpenClipArt-Vectors [Web log post - Animation]. https://pixabay.com/en/avocado-food-green-half-nutrition-161822/
	Tookapic [Web log post - Photograph]. https://pixabay.com/en/photos/avocado/?
	FoodieFactor [Web log post - Photograph]. https://pixabay.com/en/avocado-avocados-food-healthy-food-2644150/
	http://www.hassavocadoboard.com/retail/volume-and-price-data

## Project 3: NYC Taxi data - Plotting very large datasets meaningfully using datashader

	Below is the link to the reference python notebook.
https://anaconda.org/jbednar/nyc_taxi/notebook
	After downloading the notebook and reading about the exercise, I downloaded the file with all locations of all NYC taxi pickups and dropoffs from the month of March 2016 from the below link. It had over 12 million entries.
 http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml 
	Few columns from the original file were imported into a dataframe using pandas module. Initially, there were only latitude and longitude columns in it as shown below.
	To map the co-ordinate information in a map, I had to convert latitude and longitude to web Mercator co-ordinates (used in google maps and other web applications). This turned out to be a time-consuming endeavor as there are many confusing co-ordinate systems.
I had to create a user-defined function, after filtering out the null values in the data. Math turned out to be best package for this job as other packages such as pyproj and utm took hours to process 12 million entries. This piece of code was missing in the original notebook.

	1000-point scatterplot - It was interesting plotting with 1000 datapoints in the sample. As only 0.000083 %of the data from 12 million entries were represented, it lead to a phenomenon called undersampling shown in Figure (a).
	10,000-point scatterplot – lead to a phenomenon called over-sampling and it was hard to view along with the map.
	100,000-point scatterplot – lead to saturation as the points became tiny and visible at popular locations.  
	10 million-point datashaded plots - with datashader, there comes features such as auto-ranging and we can plot large datasets without over-sampling, undersampling or saturation. Below are some of the plots of 10 million datapoints with varying range and parameters. They are generated in few seconds.

	Interactive datashaded plots – Here is the unveiling of the full strength of datashader. It is interactive showing the different levels of the large dataset.
It reveals local structures that are invisible through global view.
Furthermore, it performs automatic operations and auto-ranging and optimates our output. 
	We can further customize the plot by inserting Bokeh tile- STAMEN_TERRAIN (apparently by Stamen design) that inserts a map of the area under study / visualization.
	There were several errors during these steps due to an empty array passed from the aggregate parameter. Turned out to be incorrect x and y range in the plot definition.
	The final plot looks surreal and beautiful. The interactive features of bokeh and datashader can be used to view local structures. Even though the timing of the trip is color-coded (example: red at night, blue in the evening), there are busy hours that stand-out. 

## Project 4 - Facebook replica database using SQL + Querying using Tableau and MongoDB

### Overview

Facebook Inc. is an American social media and social networking company which was established in the year of 2004 by Mark Zuckerberg, Eduardo Saverin, Andrew McCollum, Dustin Moskovitz and Chris Hughes. It was a school-based social network in Harvard University until 2006. In 2006, Facebook opened its doors to anyone older 13 years or older in the world. Facebook has a very user-friendly interface, and anyone with basic computer knowledge can use Facebook.  The primary purpose of Facebook was to find friends who have not been in touch and to help them re-connect. Among the many other social networking sites, Facebook emerged to be the most successful one due to its unique features such as the ‘Like’ option, News Feed, Games and Business-friendly approach. The usage of Facebook has grown over time, the number of users crossing 2.27 billion monthly active users, as of September 2018, according to statistics.  In the last decade, the use of all social networking sites has grown exponentially, with Facebook leading the list. This exponential growth means there is a huge amount of data available from all these users. Facebook was built around Big Data from its beginning, data was the driving force that kept it alive. But recently, Facebook has run into a lot of trouble for its usage of user data. Some attackers accessed personal data of at least 50 million Facebook users by exploiting a vulnerability in the system. This led to a huge uproar about the data usage of social networks, and increased privacy concerns among users, which led Facebook to shut down almost all of its open source data. In this project, we have attempted to create a database that is similar to that of Facebook, write queries to see how the database works and find some specific details using SQL queries such the most expensive game, a specific name search, etc. 

### Description of the Data Model

In order to proceed with this project, it is essential to understand how Facebook database works. Information on Facebook is mostly represented in the form of a social graph. The content is usually highly customizable based on the user’s privacy settings. Hence, the data has to be stored in its original form and then filtered when needed. Facebook uses a combination of MySQL and Memcache for its database. Every user has his/her own dedicated database. Facebook uses MySQL because of its speed and reliability.  Facebook stores friend relationships in a system called ‘Tao’ which uses MySQL. All tables have a hashed name and they are spread over a number of servers, similar to graph databases. Tao only stores the relations between entities. According to Facebook Inc., they collect information based on how a user uses their products. Information is collected from and about all computers and other devices the user uses to access Facebook, and this information is combined by them. The collected information is used to personalize features and content and to make suggestions for the user. The collected data is used to help advertisers to measure the effectiveness of their ads and services and to understand how the users interact with their services. They store the data until it is no longer necessary to provide the services or until the user deletes the account, whichever is first. While trying to understand the database structure of Facebook, we went through a number of resources and stumbled upon a resource where the class diagram was created by reverse engineering various Facebook business entities. Since only a very small portion of database details of Facebook is available as open source, we decided to create our database by reverse-engineering the the ER diagram we initially designed. Thereafter, we generated data ourselves before proceeding to integrate the data in sql. We went way beyond the presribed 5 problems/queries when we decided to tackle around 20 problems/queries. Some were routine tasks whereas others were aimed at analysing by querying the database.

## Side Project - Flag illegal listings in Paris Airbnb dataset

→ Dataset: AirBnb data pertaining to paris Paris

→ Source: http://insideairbnb.com/get-the-data.html

→ Scenario: Legal issues arising from listings exceeding the yearly 120 nights cap set by the government.

→ Task: Flagging illegal listings in Paris Airbnb dataset

The files with Airbnb listings, calender and neighborhoods are available for analysis.

### Motivation -
According to Critics, Airbnb drives up rental prices and lowers housing availability in cities such as Paris and San Francisco, forcing lawmakers to enact laws to protect the interests of local residents seeking accomodation. Failure to comply attracts heavy penalties, and any controversies triggered henceforth, can tarnish the public image of the company, and negatively impact customer affinity.

### Context of study -
Legal issues - Airbnb has been raking hefty fines for illegal rentals across the world, forcing lawmakers to coin newer laws. Specifically, in the city of Paris, Airbnb amassed fines of over 14 million Euros, thanks to a 2018 law that renders illegal posting (above 120 nights per year) punishable at 12,500 Euros per posting.

### Assumptions -
→ As booking data is confidential, the listing was considered booked when it is unavailable based on ‘Availability_365 field (e.g. ‘False’) in Calendar file.

→ ‘Adjusted Price’ in Calendar file was considered an accurate estimator of daily income over ‘Price’, as it probably incorporates offers and demand-based Smart-pricing.

→ We are deeming listings based on a single criterion of yearly 120 nights as Illegal/Legal and chose to ignore other regulations for ease of study.

→ We considered the current dataset from Feb’19 to Jan’20 as yearly data irrespective of calendar year.

Airbnb has been heavily penalized (€14 million fine last year) for listings that exceed the 120 nights per year legal limit. In this project, a statistical study was followed by visualizations isolating the illegal entities and their corresponding revenue.

