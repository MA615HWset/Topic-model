rm(list=ls())
library(dplyr)
library(tidytext)
library(topicmodels)
library(ldatuning)
library(ggplot2)
library(wordcloud)
library(quanteda)


movie_data <- read.csv("E:/Desktop/BU/2024 Fall/MA615/Topic/movie_plots_with_genres.csv")

movie_data_clean <- movie_data %>%
  unnest_tokens(word, Plot) %>%
  anti_join(stop_words) %>%
  filter(!word %in% c("movie", "film")) # Add more domain-specific stopwords if needed

# Create a DTM
dtm <- movie_data_clean %>%
  count(row, word) %>%
  cast_dtm(row, word, n)

# Testing a range of topic numbers from 2 to 20
lda_results <- FindTopicsNumber(
  dtm,
  topics = seq(2, 20, by = 1),
  metrics = c("CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 1234)
)


# Plot the results to find the optimal number of topics
FindTopicsNumber_plot(lda_results)

movie_data_tfidf <- movie_data_clean %>%
  count(row, word) %>%
  bind_tf_idf(word, row, n) %>%
  cast_dtm(row, word, tf_idf)

#######

optimal_k <- 7
lda_model <- LDA(dtm, k = optimal_k, method = "Gibbs", control = list(seed = 1234))

# Extract topic distributions for each document
topic_distributions <- posterior(lda_model)$topics

k_means <- kmeans(topic_distributions, centers = 5) # Experiment with different numbers of clusters

# Add cluster assignments to your dataset
movie_data$cluster <- as.factor(k_means$cluster)

pca_result <- prcomp(topic_distributions, scale = TRUE)
pca_data <- as.data.frame(pca_result$x[, 1:2]) # Use first two principal components
pca_data$cluster <- movie_data$cluster

# Plot clusters
ggplot(pca_data, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(alpha = 0.7) +
  labs(title = "Movie Topics Clustered by Similarity", x = "PC1", y = "PC2") +
  theme_minimal()

####################

lda_terms <- tidy(lda_model, matrix = "beta")

# Top terms for each topic
top_terms <- lda_terms %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

# top_terms


######## word cloud #######
for (i in 1:optimal_k) {
  wordcloud(words = filter(top_terms, topic == i)$term,
            freq = filter(top_terms, topic == i)$beta,
            scale = c(4, 0.5),
            random.order = FALSE,
            colors = brewer.pal(8, "Dark2"))
}
