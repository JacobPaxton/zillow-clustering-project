This repository contains my work for the Codeup Zillow Clustering project.

# Curriculum Rubric
## Overall
1. A clearly named final notebook. This notebook will be what you present and should contain plenty of markdown documentation and cleaned up code.
2. A README that explains what the project is, how to reproduce you work, and your notes from project planning.
3. A Python module or modules that automate the data acquisistion and preparation process. These modules should be imported and used in your final notebook.
## Specific
1. Data Acquisition: Data is collected from the codeup cloud database with an appropriate SQL query
2. Data Prep: Column data types are appropriate for the data they contain
3. Data Prep: Missing values are investigated and handled
4. Data Prep: Outliers are investigated and handled
5. Exploration: the interaction between independent variables and the target variable is explored using visualization and statistical testing
6. Exploration: Clustering is used to explore the data. A conclusion, supported by statistical testing and visualization, is drawn on whether or not the clusters are helpful/useful. At least 3 combinations of features for clustering should be tried.
7. Modeling: At least 4 different models are created and their performance is compared. One model is the distinct combination of algorithm, hyperparameters, and features.
8. Best practices on data splitting are followed
9. The final notebook has a good title and the documentation within is sufficiently explanatory and of high quality
10. Decisions and judment calls are made and explained/documented
11. All python code is of high quality

# Slack Rubric
1. make trello board public and put the link in your readme
2. 4 minute presentations
3. you will be presenting a final cleaned up noteboook
4. **you will use at least one clustering algorithm on 3+ variables to create clusters during exploration and you will explore those clusters.**
5. you will explore drivers as usual also through visualization and statistical testing
6. you will use regression modeling to find the features that are biggest drivers
7. you will have an intro, summary, conclusion in your jupyter notebook.
8. you will include key takeaways, next steps, recommendations.
9. you will use much of the work you have already done in exercises and previous project!
10. you will explore by asking questions and answering through viz and tests

# Plan
1. Acquire 'zillow' data

# Ideas:
- is_coastline (for each latitude, westmost 5 properties)
    * Visualize as a 4th hue for coordinate plot, each county + is_coastline
- is_highvalue (against raw home value)
    * Establish what correlates with home value, then look for things that should be low value but still yields high value
    * EX: High-value, small-lot
    * EX: High-value, low-bednbath
- is_highvaluecity or is_highvalueneighborhood (regioncityid, high-value cities/neighborhoods on the list)
    * Group by city ID, show average taxvaluedollarcnt, split into "high-cost", "medium cost", "low cost" groupings, use these as a feature