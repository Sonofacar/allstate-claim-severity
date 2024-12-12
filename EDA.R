library(vroom)
library(tidyverse)
library(patchwork)

train <- vroom("train.csv")
test <- vroom("test.csv")
data <- train[1:132]
tmp <- test
tmp$loss <- NA
tmp$type <- "test"
data$type <- "train"
data <- rbind(data, tmp)

categories <- train[2:117]
conts <- train[118:131]
response <- train[132]

for (col in colnames(categories)) {
  train[col] <- train[[col]] %>% factor
}

categories <- train[2:117]

# Getting basic info about Categorical variables
cat_info <- tibble(Column = character(),
                   Category = character(),
                   Count = integer(),
                   Pred_count = integer(),
                   Count_diff = integer(),
                   Weakness_ratio = double(),
                   Average = double(),
                   Std_dev = double())
i <- 1
for (col in colnames(categories)) {
  cats <- categories[[col]] %>%
    levels
  for (level in cats) {
    avg <- mean(train[train[[col]] == level, ][["loss"]])
    std_dev <- sd(train[train[[col]] == level, ][["loss"]])
    num <- length(train[train[[col]] == level, ][[col]])
    pred_num <- length(test[test[[col]] == level, ][[col]])
    diff <- abs(num - pred_num)
    ratio <- diff / num
    cat_info[i, ] <- list(Column = col,
                          Category = level,
                          Count = num,
                          Pred_count = pred_num,
                          Count_diff = diff,
                          Weakness_ratio = ratio,
                          Average = avg,
                          Variance = std_dev)
    i <- i + 1
  }
}
cat_info[is.na(cat_info$Std_dev), "Std_dev"] <- 0

# Finding categories that don't matter
no_matter <- cat_info[cat_info$Pred_count == 0, ]
write("Potentially remove...", stdout())
for (i in seq_along(no_matter[["Column"]])) {
  paste(no_matter[i, "Column"], ": ", no_matter[i, "Category"], sep = "") %>%
    write(stdout())
}

# make plots
cols <- cat_info$Column %>% unique
for (i in seq_along(cols)) {
  col <- cols[i]
  if ((i + 1) < 116) {
    for (comp in cols[(i + 1):116]) {
      name1 <- sym(col)
      name2 <- sym(comp)
      counts_plot <- ggplot(data) +
        geom_bar(aes(x = {{name1}}, fill = {{name2}}), position = "dodge") +
        facet_wrap(vars(type)) +
        theme_classic()
      box_plots <- ggplot(data[data$type == "train", ]) +
        geom_boxplot(aes(x = loss, y = {{name1}}, fill = {{name2}})) +
        scale_x_continuous(trans = "log10") +
        theme_classic()
      complete_plot <- counts_plot / box_plots
      paste("graphs/basic/", col, "x", comp, ".png", sep = "") %>%
        png()
      print(complete_plot)
      dev.off()
    }
  }
}

for (cat in cols) {
  for (cont in colnames(conts)) {
    name1 <- sym(cat)
    name2 <- sym(cont)
    plot <- ggplot(data) +
      geom_point(aes(x = {{name2}},
                     y = loss,
                     shape = {{name1}},
                     color = {{name1}}),
                 alpha = 0.3) +
      theme_classic()
    paste("graphs/basic/", cont, "x", cat, ".png", sep = "") %>%
      png()
    print(plot)
    dev.off()
  }
}

# interaction plots
cols <- cat_info$Column %>% unique
for (i in seq_along(cols)) {
  col <- cols[i]
  if ((i + 1) < 116) {
    for (comp in cols[(i + 1):116]) {
      name1 <- sym(col)
      name2 <- sym(comp)
      plot <- train %>%
        select({{name1}}, {{name2}}, loss) %>%
        group_by({{name1}}, {{name2}}) %>%
        summarize(loss = mean(loss)) %>%
        ggplot() +
        geom_line(aes(x = {{name1}},
                      y = loss,
                      color = {{name2}},
                      group = {{name2}}))
      paste("graphs/interactions/", col, "x", comp, ".png", sep = "") %>%
        png()
      print(plot)
      dev.off()
    }
  }
}

for (cat in cols) {
  for (cont in colnames(conts)) {
    name1 <- sym(cat)
    name2 <- sym(cont)
    plot <- ggplot(data) +
      geom_point(aes(x = {{name2}},
                     y = loss,
                     shape = {{name1}},
                     color = {{name1}}),
                 alpha = 0.3) +
      theme_classic()
    paste("graphs/", cont, "x", cat, ".png", sep = "") %>%
      png()
    print(plot)
    dev.off()
  }
}


