# F1 Race Winner Prediction 🏎️

# Problem Definition:
The core objective of our study is to predict the likelihood of a Formula 1 driver winning a race. This prediction is based on a variety of features derived from historical data and race conditions.
# Hypothesis:
Our analysis operates under the hypothesis that certain factors significantly influence race outcomes. We posit that drivers who start from higher positions on the grid are more likely to win, reflecting the advantage of starting closer to the front. Additionally, historical performance metrics such as previous race positions and accumulated points are considered indicative of a driver's likelihood to succeed. These factors suggest that sustained performance and advantageous starting positions are critical determinants in a driver's probability of winning.
# Task:
Our task involves developing a predictive model that utilizes these insights to forecast race outcomes effectively. The model will employ machine learning techniques to analyze patterns and relationships within the data, focusing particularly on how starting grid positions and past performance metrics influence a driver's chances of winning.
# Target Variable:
The target variable for our model is 'Winner,' a binary label that identifies whether a driver won a particular race. This label serves as the primary outcome around which our predictive modeling strategies are centered.

# Dataset Overview
Our dataset encapsulates a vast array of historical data on Formula 1, spanning from 1950 to the latest 2023 season. It includes extensive details about races, drivers, constructors, qualifying rounds, circuits, lap times, pit stops, and championship outcomes. For the purpose of this analysis, we will narrow our focus to data from the year 2000 onwards, ensuring that the insights and models developed are relevant to contemporary Formula 1 dynamics. The key datasets utilized include:
results.csv: This file records comprehensive race results for each event, detailing result ID, race ID, driver ID, constructor ID, starting grid position, final race position, points scored, laps completed, total race time, fastest lap details, and driver status.
races.csv: Provides specific information about each race, such as race ID, year, round, circuit ID, race name, date, and time. Additional details include URLs for further information and timestamps for practice sessions, qualifying rounds, and sprint races, offering a chronological overview of each race event.
drivers.csv: Contains personal and professional information about the drivers, including driver ID, reference number, race number, code, full name, date of birth, nationality, and a URL for detailed profiles, enriching the dataset with personal insights and career trajectories of the drivers.
constructors.csv: Details the teams involved in constructing the race cars, listing constructor ID, reference number, team name, nationality, and URL, providing context on the teams' backgrounds and contributions to the sport.
These datasets were meticulously merged based on common identifiers such as "raceId," "driverId," and "constructorId" to create a unified DataFrame, "merged_df." This integration facilitates a comprehensive analysis and modeling, allowing us to delve into the intricacies and outcomes of Formula 1 races in a structured and insightful manner.

# Preprocessing
This part of the code performs preprocessing tasks on the Formula 1 dataset before splitting it into training and testing sets. Here's what it does:

Data Loading and Merging:
We initiated the process by loading data from multiple CSV files into separate DataFrames: races_df, results_df, drivers_df, and constructors_df. Notably, we handled the '\N' values, recognizing them as NaN during the loading process to ensure data integrity. Subsequently, we merged these individual DataFrames into a cohesive single DataFrame named merged_df, enabling comprehensive analysis and modeling on the combined dataset.

Handling Missing Values:
Missing values are common in real-world datasets and can adversely affect the performance of machine learning models. In this preprocessing step, missing values in both numeric and categorical columns are handled by imputing them with appropriate values. Numeric columns are imputed with the median value, while categorical columns are imputed with the mode value.

Encoding Categorical Variables:
Machine learning models require numerical input, so categorical variables need to be converted into a numerical format. This process, known as one-hot encoding, creates dummy variables for each category within a categorical feature. It allows the model to interpret and use categorical data effectively during training.

Feature Engineering:
Feature engineering involves creating new features from existing ones to improve the predictive power of the model. In this code, several new features are created based on domain knowledge and insights. These include:
Average finish position for each driver: Provides insight into a driver's overall performance.
Winner column: Indicates whether a driver won the race (binary classification).
Win rate for each constructor: Reflects the success rate of a constructor in winning races.
Average qualifying position for each driver: Provides insight into a driver's performance in qualifying sessions.
Driver's age at the time of the race: Captures the impact of age on performance.
Experience level: Represents the number of years since a driver's first race, indicating their level of experience in Formula 1.
Extracting month from the race date: Allows the model to capture any seasonal patterns or trends in race outcomes.

Handling Class Imbalance:
Class imbalance occurs when one class (e.g., winners) is significantly more prevalent than the other class (e.g., non-winners) in the dataset. This can bias the model towards the majority class and lead to poor performance. To address this issue, oversampling of the minority class (winners) is performed to balance the class distribution, ensuring that both classes are equally represented in the training data.

Dimensionality Reduction (PCA):
High-dimensional datasets may suffer from the curse of dimensionality, leading to increased computational complexity and potential overfitting. Principal Component Analysis (PCA) is a feature selection technique used to reduce the dimensionality of the feature space while preserving most of the information. It transforms the original features into a new set of orthogonal variables (principal components) that capture the maximum variance in the data. In this code, PCA is applied to reduce the dimensionality of the feature space to five principal components, which helps simplify the model and improve its generalization performance.

Random Sampling for Train-Test Split:
Before training a machine learning model, it's essential to split the dataset into separate training and testing sets to evaluate the model's performance on unseen data. Random sampling is used to randomly select a subset of the data for both training and testing, ensuring that the data is representative and unbiased. In this code, 30% of the data is randomly sampled for both the training and testing sets, with a fixed random seed for reproducibility. The following output illustrates the dimensions of both the training and testing datasets:
Training set (features): 3761 samples, each with 5 features.
Testing set (features): 1612 samples, each with 5 features.
Training set (labels): 3761 samples.
Testing set (labels): 1612 samples.

# Model Building
First, we prepare the data by creating a DataFrame from the training set. Each feature is named sequentially from 'Feature_1' to 'Feature_n' for clarity. A target column 'winner' is added, which holds the labels from our training set.

We then initialize PyCaret with the setup function, passing our training DataFrame and specifying the target column. Here, we choose to normalize the data, remove multicollinear features (those with a correlation above 95%), and set other configurations to streamline the process, such as turning off preprocessing since it’s manually handled and muting verbose output for cleaner execution.

Using PyCaret’s compare_models function, we select the top three models to proceed with, specifically including decision trees ('dt'), logistic regression ('lr'), and random forests ('rf'). This function compares several models based on default metrics and returns the best performing models as per the given criteria.

# Model Evaluation and Tuning
Each selected model is then evaluated and tuned:

Initial Evaluation: We use evaluate_model to observe the performance of each model before tuning. This function provides a visual and statistical review of various metrics.
Tuning: For each model, we define a set of hyperparameters to explore. For instance, we vary 'max_depth' for the decision tree, regularization strength 'C' for logistic regression, and 'n_estimators' for the random forest. Using tune_model, we adjust these hyperparameters one at a time to see if the performance improves. This function allows specifying a custom grid for each parameter, providing a focused and efficient tuning process.
After tuning, we re-evaluate the models to pull the latest performance metrics and store these in a dictionary. This systematic approach helps in comparing the original and tuned models' performance directly.

Finally, we print the results for each model, both pre and post-tuning, providing a clear, structured output of the models' performances across different settings. This process is instrumental in identifying the best model configuration for the prediction task based on the provided training data.






