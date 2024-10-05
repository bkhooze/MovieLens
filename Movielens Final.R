library(tidyverse)
library(caret)
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip
options(timeout = 120)
dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings_file <- "ml-10M100K/ratings.dat"
unzip(dl, ratings_file)
movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))
movielens <- left_join(ratings, movies, by = "movieId")
# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)
# Installing package format R to keep code tidy
install.packages("formatR", repos = "http://cran.us.r-project.org")
library(formatR)
edx %>% as_tibble()
# The dataset consists of 9,000,055 rows and 6 columns
edx %>% summarise_all(~ sum(is.na(.)))
# No NA values are noted in the dataset
edx %>% group_by(title) %>% summarize (total_ratings = n())
edx %>% group_by(rating) %>% summarize(count = n()) %>% arrange(desc(count))
# This table shows the movies with the highest ratings.
edx %>% group_by(movieId) %>% ggplot(aes(rating)) + geom_histogram(binwidth = 0.5, fill = "yellow", col = "black") + ggtitle("Histogram for Rating Distribution") + ylab("Number of Ratings") + xlab("Rating")
library(lubridate)
edx <- edx %>% mutate(year_rating = as_datetime(edx$timestamp))
edx <- edx %>% mutate(year_rating = year(year_rating))
# Extract movie year and calculate movie age at time of rating
edx <- edx %>% mutate(year_movie = substr(edx$title, nchar(edx$title) - 4, nchar(edx$title) - 1))
range(edx$year_movie)
edx <- edx %>% mutate(age_movie = year_rating - as.numeric(year_movie))
# As year of rating cannot precede year of movie, negative values of movie age at year of rating were set to 0.
range(edx$age_movie)
edx <- edx %>% mutate(age_movie = ifelse(age_movie<0, 0, age_movie))
edx %>% ggplot(aes(age_movie)) + 
  geom_histogram(binwidth = 0.5, fill = "yellow", col = "black") + ggtitle("Number of Movies by Movie Age") + ylab("Number of Ratings") + xlab("Age of Movie at Time of Rating")
# Partition edx set to test and train set for model development
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-test_index,]
edx_test <- edx[test_index,]
# Define root mean square error, a common measure used to determine model performance in machine learning.
RMSE <- function(actual_rating, predicted_rating){
  sqrt(mean((actual_rating - predicted_rating)^2))
}
# Defining the average, mu, using the training set and comparing this to the test set
mu <- mean(edx_train$rating)
naive_RMSE <- RMSE(edx_test$rating, mu)
naive_RMSE
# Model 1: Average movie rating effects, then remove NA values
average_movie_rating <- edx_train %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu))
average_movie_rating %>% ggplot(aes(b_i)) + geom_histogram(binwidth = 0.2, fill = "yellow", col = "black") + ggtitle("Distribution of Movie Effects") + ylab("Count") + xlab("Difference between actual movie rating and average movie rating")
model_1 <- edx_test %>% left_join(average_movie_rating, by = 'movieId') %>% mutate(predicted_rating = mu + b_i) %>% pull(predicted_rating)
RMSE_AMR <- RMSE(edx_test$rating, model_1)
RMSE_AMR
sum(is.na(model_1))
# Noted 17 NAs were generated during the join procedure and these are removed.
summary(edx_test %>% left_join(average_movie_rating, by = 'movieId') %>% pull(b_i))
model_1[is.na(model_1)] <- mu
RMSE_model_1 <- RMSE(edx_test$rating, model_1)
RMSE_model_1
# Model 2: Average movie rating effects and user effects
user_average_movie_rating <- edx_train %>% left_join(average_movie_rating, by = 'movieId') %>% group_by(userId) %>% summarize(b_u = mean(rating - mu - b_i))
user_average_movie_rating %>% ggplot(aes(b_u)) + geom_histogram(binwidth = 0.2, fill = "yellow", col = "black") + ggtitle("Distribution of User Effects") + ylab("Count") + xlab("User Effects")
model_2 <- edx_test %>% left_join(average_movie_rating, by = 'movieId') %>% left_join(user_average_movie_rating, by = 'userId') %>% mutate(predicted_rating = mu + b_i + b_u) %>% pull(predicted_rating)
model_2[is.na(model_2)] <- mu
RMSE_model_2 <- RMSE(edx_test$rating, model_2)
RMSE_model_2
# Model 3: Average movie rating effects + user effects + age of movie at time of rating effects
age_average_movie_rating <- edx_train %>% left_join(average_movie_rating, by = 'movieId') %>% left_join(user_average_movie_rating, by = 'userId') %>% group_by(age_movie) %>% summarize(b_a = mean(rating - mu - b_i - b_u))
age_average_movie_rating %>% ggplot(aes(b_a)) + geom_histogram(binwidth = 0.01, fill = "yellow", col = "black") + ggtitle("Distribution of Age of Movie at Time of Rating Effects") + ylab("Percentage of Observations") + xlab("Age of Movie at Time of Rating")
model_3 <- edx_test %>% left_join(average_movie_rating, by = 'movieId') %>% left_join(user_average_movie_rating, by = 'userId') %>% left_join(age_average_movie_rating, by = 'age_movie') %>% mutate(predicted_rating = mu + b_i + b_u + b_a) %>% pull(predicted_rating)
model_3[is.na(model_3)] <- mu
RMSE_model_3 <- RMSE(edx_test$rating, model_3)
RMSE_model_3
# Model 4: Adding regularisation to Model 2 and finding the optimal lambda
lambda <- seq(0,10,1)
RMSES <- sapply (lambda, function(l){
  mu <- mean(edx_train$rating)
  b_i <- edx_train %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n() + l))
  b_u <- edx_train %>% left_join(b_i, by = 'movieId') %>% group_by(userId) %>% summarize(b_u = sum(rating - mu - b_i)/(n() + l))
  predicted_rating <- edx_test %>% left_join(b_i, by = 'movieId') %>% left_join(b_u, by = 'userId') %>% mutate(pred = mu + b_i + b_u) %>% pull(pred)
  predicted_rating[is.na(predicted_rating)] <- mu
  return(RMSE(edx_test$rating, predicted_rating))
})
qplot(lambda,RMSES)
RMSE_model_4 <- min(RMSES)
# Model 5: Adding regularisation to Model 3 (average movie rating effects + user effects + age of movie at time of rating effects)
lambda <- seq(0,10,1)
RMSES_1 <- sapply (lambda, function(l){
  mu <- mean(edx_train$rating)
  b_i <- edx_train %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n() + l))
  b_u <- edx_train %>% left_join(b_i, by = 'movieId') %>% group_by(userId) %>% summarize(b_u = sum(rating - mu - b_i)/(n() + l))
  b_a <- edx_train %>% left_join(b_i, by = 'movieId') %>% left_join(b_u, by = 'userId') %>% group_by(age_movie) %>% summarize(b_a = sum(rating - mu - b_i - b_u)/(n() + l))
  predicted_rating <- edx_test %>% left_join(b_i, by = 'movieId') %>% left_join(b_u, by = 'userId') %>% left_join(b_a, by = 'age_movie') %>% mutate(pred = mu + b_i + b_u + b_a) %>% pull(pred)
  predicted_rating[is.na(predicted_rating)] <- mu
  return(RMSE(edx_test$rating, predicted_rating))
})
qplot(lambda,RMSES_1)
lambda <- 5
b_i <- edx_train %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n() + lambda))
b_u <- edx_train %>% left_join(b_i, by = 'movieId') %>% group_by(userId) %>% summarize(b_u = sum(rating - mu - b_i)/(n() + lambda))
b_a <- edx_train %>% left_join(b_i, by = 'movieId') %>% left_join(b_u, by = 'userId') %>% group_by(age_movie) %>% summarize(b_a = sum(rating - mu - b_i - b_u)/(n() + lambda))
predicted_rating <- edx_test %>% left_join(b_i, by = 'movieId') %>% left_join(b_u, by = 'userId') %>% left_join(b_a, by = 'age_movie') %>% mutate(pred = mu + b_i + b_u + b_a) %>% pull(pred)
predicted_rating[is.na(predicted_rating)] <- mu
RMSE_model_5 <- RMSE(edx_test$rating, predicted_rating)
# Mutate final_holdout_test to include age_movie column, set negative values to 0
final_holdout_test <- final_holdout_test %>% mutate(year_rating = as_datetime(final_holdout_test$timestamp))
final_holdout_test <- final_holdout_test %>% mutate(year_rating = year(year_rating))
final_holdout_test <- final_holdout_test %>% mutate(year_movie = substr(final_holdout_test$title, nchar(final_holdout_test$title) - 4, nchar(final_holdout_test$title) - 1))
final_holdout_test <- final_holdout_test %>% mutate(age_movie = year_rating - as.numeric(year_movie))
final_holdout_test <- final_holdout_test %>% mutate(age_movie = ifelse(age_movie<0, 0, age_movie))
lambda <- 5
# Using model 5 that was previously derived from the edx train set to test the RMSE on the final holdout test set
final_mu <- mean(edx$rating)
final_b_i <- edx %>% group_by(movieId) %>% summarize(final_b_i = sum(rating - final_mu)/(n() + lambda))
final_b_u <- edx %>% left_join(final_b_i, by = 'movieId') %>% group_by(userId) %>% summarize(final_b_u = sum(rating - final_mu - final_b_i)/(n() + lambda))
final_b_a <- edx %>% left_join(final_b_i, by = 'movieId') %>% left_join(final_b_u, by = 'userId') %>% group_by(age_movie) %>% summarize(final_b_a = sum(rating - final_mu - final_b_i - final_b_u)/(n() + lambda))
Final_model_5 <- final_holdout_test %>% left_join(final_b_i, by = 'movieId') %>% left_join(final_b_u, by = 'userId') %>% left_join(final_b_a, by = 'age_movie') %>% mutate(pred = final_mu + final_b_i + final_b_u + final_b_a) %>% pull(pred)
Final_model_5[is.na(predicted_rating)] <- final_mu
Final_RMSE_model_5 <- RMSE(final_holdout_test$rating, Final_model_5)
Final_RMSE_model_5
# Summarising results obtained in a table
results <- tibble(Model_Type = c("Baseline", "Training", "Training", "Training",
                                 "Training", "Training", "Validation"),
                  Model = c("Baseline RMSE", "Average Rating Effects",
                            "Average Rating + User Effects",
                            "Average Rating + User + Age Effects",
                            "Average Rating + User + Regularization", 
                            "Average Rating + User + Age + Regularization",
                            "Final RMSE on Final Test Set"),
                  RMSE_Calculated = c(naive_RMSE, RMSE_model_1, RMSE_model_2,
                                      RMSE_model_3, RMSE_model_4,
                                      RMSE_model_5, Final_RMSE_model_5))
install.packages("tibble", repos = "http://cran.us.r-project.org")
library(tibble)
options(pillar.sigfig = 5)
results
