library(jsonlite)
library(tidyverse)
library(stringr)
library(e1071)
library(dplyr)
#Datset pipeline
files <- list.files("sample_data/", full.names = TRUE)
parse_post <- function(file) {
  tryCatch({
    data <- fromJSON(file, simplifyVector = FALSE)  # IMPORTANT
    
    # ---- CAPTION EXTRACTION ----
    caption_text <- NA
    
    if (!is.null(data$edge_media_to_caption$edges) &&
        length(data$edge_media_to_caption$edges) > 0) {
      
      edge <- data$edge_media_to_caption$edges[[1]]
      
      if (!is.null(edge$node$text)) {
        caption_text <- edge$node$text
      }
    }
    
    # ---- COMMENTS ----
    comments_count <- NA
    if (!is.null(data$edge_media_to_comment$count)) {
      comments_count <- data$edge_media_to_comment$count
    } else if (!is.null(data$edge_media_preview_comment$count)) {
      comments_count <- data$edge_media_preview_comment$count
    }
    
    tibble(
      username = data$owner$username,
      likes = as.numeric(data$edge_media_preview_like$count),
      comments = as.numeric(comments_count),
      caption = caption_text,
      timestamp = data$taken_at_timestamp
    )
    
  }, error = function(e) {
    return(NULL)
  })
}

test <- map_dfr(files[1:50], parse_post)
print(test)


batch_size <- 2000
results <- list()

for (i in seq(1, length(files), by = batch_size)) {
  batch_files <- files[i:min(i + batch_size - 1, length(files))]
  batch_data <- map_dfr(batch_files, parse_post)
  results[[length(results) + 1]] <- batch_data
  print(paste("Processed:", i))
}
posts_df <- bind_rows(results)
posts_df <- posts_df %>%
  mutate(
    hashtags = str_extract_all(caption, "#\\w+"),
    hashtag_count = lengths(hashtags)
  )
write.csv(posts_df, "processed_data/posts_clean.csv", row.names = FALSE)

file <- files[1]
readLines(file, n = 50)


posts_df <- posts_df %>%
  mutate(
    hashtags_str = sapply(hashtags, function(x) paste(x, collapse = " "))
  )
saveRDS(posts_df, "processed_data/posts_clean.rds")

posts_df <- posts_df %>%
  mutate(
    hashtags_str = sapply(hashtags, function(x) paste(x, collapse = " "))
  ) %>%
  select(-hashtags)
sapply(posts_df,class)

df <- read.csv("processed_data/posts_clean.csv")
str(df)
dim(df)
colSums(is.na(df))
head(df)
influencers <- read.delim("influencers.txt", header = FALSE,stringsAsFactors = FALSE)
str(influencers)
influencers <- influencers[-c(1,2), ]
colnames(influencers) <- c("username", "category", "followers", "following", "posts")
influencers$followers <- as.numeric(influencers$followers)
influencers$following <- as.numeric(influencers$following)
influencers$posts <- as.numeric(influencers$posts)
str(influencers)
head(influencers)
posts_df <- left_join(posts_df, influencers, by = "username")
colSums(is.na(posts_df))
sum(!is.na(posts_df$followers))
posts_df <- posts_df %>%
  filter(!is.na(followers))


#Derived features
posts_df <- posts_df %>%
  mutate(
    caption_length = nchar(caption),
    emoji_count = str_count(caption, "[^\\w\\s,]")
  )
#Target
median_likes <- median(posts_df$likes)
posts_df <- posts_df %>%
  mutate(
    high_engagement = ifelse(likes > median_likes, 1, 0)
  )
saveRDS(posts_df, "processed_data/posts_final.rds")
write.csv(posts_df, "processed_data/posts_final.csv", row.names = FALSE)
save(posts_df, file = "processed_data/posts_final.RData")
test <- readRDS("processed_data/posts_final.rds")
dim(test)

#hastgas feature engineering
all_tags <- unlist(strsplit(posts_df$hashtags_str, " "))
all_tags <- all_tags[all_tags != ""]
tag_freq <- sort(table(all_tags), decreasing = TRUE)
head(tag_freq, 20)
top_tags <- names(tag_freq)[1:500]
create_tag_features <- function(tags, top_tags) {
  tag_list <- strsplit(tags, " ")
  
  sapply(top_tags, function(tag) {
    sapply(tag_list, function(x) as.integer(tag %in% x))
  })
}
tag_matrix <- create_tag_features(posts_df$hashtags_str, top_tags)
tag_df <- as.data.frame(tag_matrix)
colnames(tag_df) <- make.names(top_tags)
final_df <- cbind(posts_df, tag_df)
dim(final_df)


#PCA
tag_only <- final_df[, (ncol(posts_df)+1):ncol(final_df)]
tag_matrix <- as.matrix(tag_only)
pca_result <- prcomp(tag_matrix, scale. = TRUE)
summary(pca_result)
plot(pca_result, type = "l")
sum(is.na(tag_matrix))

pca_features <- pca_result$x[, 1:15]
dim(pca_features)

#Clustering
pca_data <- pca_result$x[, 1:15]
wss <- numeric(10)
for (k in 2:10) {
  km <- kmeans(pca_data, centers = k, nstart = 10)
  wss[k] <- km$tot.withinss
}
plot(2:10, wss[2:10], type = "b", xlab = "k", ylab = "WSS")


set.seed(42)
kmeans_result <- kmeans(pca_data, centers = 4, nstart = 20)
final_df$cluster <- kmeans_result$cluster
kmeans_result

median_likes <- median(final_df$likes)
final_df$high_engagement <- ifelse(final_df$likes > median_likes, 1, 0)
model_data <- final_df[, c(
  "high_engagement",
  "followers", "following", "posts",
  "hashtag_count", "caption_length",
  paste0("PC", 1:15),
  "cluster"
)]
setdiff(c(
  "high_engagement","followers","following","posts",
  
  "hashtag_count","caption_length",
  paste0("PC",1:15),"cluster"
))
colnames(final_df)
pca_features <- pca_result$x[, 1:15]
colnames(pca_features) <- paste0("PC", 1:15)
final_df <- cbind(final_df, pca_features)
colnames(final_df)
final_df_model <- final_df[, c(
  "high_engagement",
  "followers", "following", "posts",
  "hashtag_count", "caption_length",
  paste0("PC", 1:15),
  "cluster"
)]
model_data <- final_df_model




#Test-Train Split
set.seed(42)
train_idx <- sample(1:nrow(model_data), 0.8*nrow(model_data))
train <- model_data[train_idx, ]
test <- model_data[-train_idx, ]


#logistic regression
model_logit <- glm(high_engagement ~ ., data = train, family = "binomial")
pred_prob <- predict(model_logit, test, type = "response")
pred_class <- ifelse(pred_prob > 0.5, 1, 0)
mean(pred_class == test$high_engagement)

sum(is.na(pred_class))
sum(is.na(test$high_engagement))
mean(pred_class == test$high_engagement, na.rm = TRUE)

pred_prob[is.na(pred_prob)] <- 0.5
pred_class <- ifelse(pred_prob > 0.5, 1, 0)
mean(pred_class == test$high_engagement)

#SVM
train$caption_length[is.na(train$caption_length)] <- 0
test$caption_length[is.na(test$caption_length)] <- 0
train_scaled <- train
test_scaled <- test
num_cols <- colnames(train_scaled)[-1]
train_means <- apply(train_scaled[, num_cols], 2, mean)
train_sds <- apply(train_scaled[, num_cols], 2, sd)
train_scaled[, num_cols] <- scale(train_scaled[, num_cols],
                                  center = train_means,
                                  scale = train_sds)

test_scaled[, num_cols] <- scale(test_scaled[, num_cols],
                                 center = train_means,
                                 scale = train_sds)
colSums(is.na(train_scaled))
model_svm <- svm(as.factor(high_engagement) ~ ., data = train_scaled)
pred_svm <- predict(model_svm, test_scaled)
mean(pred_svm == as.factor(test_scaled$high_engagement))


#Apriori
high_posts <- posts_df[posts_df$likes > median(posts_df$likes), ]
tag_list <- strsplit(high_posts$hashtags_str, " ")
top_tags <- names(tag_freq)[1:50]
filtered_tags <- lapply(tag_list, function(x) x[x %in% top_tags])

pairs <- list()
for (tags in filtered_tags) {
  if (length(tags) > 1) {
    comb <- t(combn(tags, 2))
    pairs <- rbind(pairs, comb)
  }
}
pairs_df <- as.data.frame(pairs)
colnames(pairs_df) <- c("A", "B")

pair_counts <- pairs_df %>%
  group_by(A, B) %>%
  summarise(freq = n()) %>%
  arrange(desc(freq))

head(pair_counts, 10)
#Confidence
tag_counts <- table(unlist(filtered_tags))
pair_counts$A <- as.character(pair_counts$A)  #A and B to character
pair_counts$B <- as.character(pair_counts$B)
pair_counts <- pair_counts %>%
  mutate(
    conf_A_to_B = freq / tag_counts[A],
    conf_B_to_A = freq / tag_counts[B]
  )
pair_counts %>%
  arrange(desc(conf_A_to_B)) %>%
  head(10)

saveRDS(final_df, "processed_data/final_with_pca_cluster.rds")
write.csv(final_df, "processed_data/final_with_pca_cluster.csv", row.names = FALSE)

saveRDS(model_data, "processed_data/model_data.rds")
write.csv(model_data, "processed_data/model_data.csv", row.names = FALSE)
writeLines(capture.output(sessionInfo()), "session_info.txt")
saveRDS(pca_result, "processed_data/pca_result.rds")
saveRDS(kmeans_result, "processed_data/kmeans_result.rds")
saveRDS(tag_freq, "processed_data/tag_frequency.rds")
