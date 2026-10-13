# ApexReality — Fake News Detection (R Shiny)

A Shiny web app that classifies news text as **FAKE** or **REAL**, with
visual analytics (confidence gauge, matched keywords, word clouds, and
frequency comparisons), built around the ApexReality logo.

## Project structure
```
apexreality/
├── app.R                  # Shiny app (UI + server + model)
├── data/
│   └── news_dataset.csv   # Sample training data (text,label)
└── www/
    └── logo.png           # ApexReality logo (shown in header)
```

## Run it locally

1. Install R (4.x) and RStudio (optional but convenient).
2. Open `app.R` in RStudio, or run from a terminal:
   ```r
   setwd("apexreality")
   shiny::runApp()
   ```
   The script auto-installs any missing packages on first run:
   `shiny, shinythemes, tm, SnowballC, wordcloud, RColorBrewer, ggplot2, plotly, e1071, DT, scales, caret`.

3. The app opens in your browser. Go to the **Analyze** tab, paste an
   article (and optional headline), pick a source/category, and click
   **Analyze Article**. Then check the **Visual Insights** tab for charts,
   and **Dataset** tab to see the training data.

## Current dataset

`data/news_dataset.csv` now contains your **500-row labeled dataset**
(252 FAKE / 248 REAL), with `title` + `text` combined into one `text`
column. The model is trained with:

- **80/20 stratified train/test split** (`caret::createDataPartition`)
  so accuracy is measured on articles the model never saw during training.
- **TF-IDF weighting** instead of raw word counts — gives more weight to
  distinctive words and less to generic filler.
- **Cached model** (`model_cache.rds`) — trains once, then reuses the
  saved model on every app restart so it doesn't retrain from scratch
  each time.

Check the **Model Performance** tab in the app for test accuracy and the
confusion matrix.

## Updating the dataset later

To retrain on a new or larger dataset:
1. Replace `data/news_dataset.csv` with a new file that has `text` and
   `label` columns (label values `FAKE`/`REAL`, case-insensitive).
2. **Delete `model_cache.rds`** (important — otherwise the app keeps
   using the old cached model).
3. Restart the app. It will retrain automatically and re-cache.

If you want to go back to the full Kaggle **"Fake and Real News Dataset"**
by Clément Bisaillon (44k articles):
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
```r
fake <- read.csv("Fake.csv"); fake$label <- "FAKE"
real <- read.csv("True.csv"); real$label <- "REAL"
fake$text <- paste(fake$title, fake$text)
real$text <- paste(real$title, real$text)
combined <- rbind(fake[, c("text","label")], real[, c("text","label")])
write.csv(combined, "data/news_dataset.csv", row.names = FALSE)
```
Then delete `model_cache.rds` and restart.

## Notes
- Model: Naive Bayes over a cleaned/stemmed bag-of-words (tm package).
  Swap in `caret`, `glmnet`, or `tidymodels` for stronger models as needed.
- Logo: `www/logo.png` is referenced directly as `src="logo.png"` — Shiny
  automatically serves anything inside `www/`.
- Deploy with shinyapps.io, Posit Connect, or Docker + Shiny Server.
