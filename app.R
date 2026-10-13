# =========================================================
# ApexReality — Fake News Detection Shiny App (UI Enhanced)
# =========================================================
# Backend (model training, prediction, caching) is unchanged.
# Only UI and visual layout have been upgraded.
# =========================================================

required_packages <- c(
  "shiny", "shinythemes", "tm", "SnowballC", "wordcloud",
  "RColorBrewer", "ggplot2", "plotly", "e1071", "DT", "scales", "caret"
)
new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(new_packages) > 0) {
  install.packages(new_packages, repos = "https://cloud.r-project.org")
}

library(shiny)
library(shinythemes)
library(tm)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)
library(ggplot2)
library(plotly)
library(e1071)
library(DT)
library(scales)
library(caret)

set.seed(42)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
dataset_path <- "data/news_dataset.csv"
news_data <- read.csv(dataset_path, stringsAsFactors = FALSE)
news_data <- news_data[!is.na(news_data$text) & nchar(trimws(news_data$text)) > 0, ]
news_data$label <- factor(toupper(trimws(news_data$label)), levels = c("FAKE", "REAL"))

# ---------------------------------------------------------
# 2. TEXT CLEANING
# ---------------------------------------------------------
clean_corpus <- function(text_vector) {
  corpus <- VCorpus(VectorSource(text_vector))
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, stemDocument)
  corpus
}

# ---------------------------------------------------------
# 3. TRAIN/TEST SPLIT (80/20, stratified) + CACHE MODEL
# ---------------------------------------------------------
cache_file <- "model_cache.rds"

train_pipeline <- function() {
  train_idx <- caret::createDataPartition(news_data$label, p = 0.8, list = FALSE)
  train_data <- news_data[train_idx, ]
  test_data  <- news_data[-train_idx, ]
  
  # --- raw term-frequency matrix on TRAIN only (this defines the vocab) ---
  train_corpus <- clean_corpus(train_data$text)
  train_dtm_tf <- DocumentTermMatrix(train_corpus, control = list(weighting = weightTf))
  train_dtm_tf <- removeSparseTerms(train_dtm_tf, 0.99)
  
  train_tf <- as.matrix(train_dtm_tf)
  colnames(train_tf) <- make.names(colnames(train_tf), unique = TRUE)
  vocab <- colnames(train_tf)
  
  # --- IDF computed ONCE from the training corpus, reused for test + live predictions.
  n_train_docs <- nrow(train_tf)
  doc_freq <- colSums(train_tf > 0)
  idf <- log(n_train_docs / (doc_freq + 1)) + 1   # smoothed idf, never zero
  names(idf) <- vocab
  
  apply_tfidf <- function(tf_matrix) {
    tf_matrix <- tf_matrix[, vocab, drop = FALSE]
    sweep(tf_matrix, 2, idf[vocab], `*`)
  }
  
  train_matrix <- as.data.frame(apply_tfidf(train_tf))
  train_matrix$label <- train_data$label
  
  nb_model <- naiveBayes(label ~ ., data = train_matrix, laplace = 1)
  
  # --- evaluate on held-out test set, using the SAME stored idf ---
  test_corpus <- clean_corpus(test_data$text)
  test_dtm_tf <- DocumentTermMatrix(test_corpus, control = list(dictionary = vocab, weighting = weightTf))
  test_tf <- as.matrix(test_dtm_tf)
  colnames(test_tf) <- make.names(colnames(test_tf), unique = TRUE)
  missing_cols <- setdiff(vocab, colnames(test_tf))
  for (col in missing_cols) {
    test_tf <- cbind(test_tf, matrix(0, nrow(test_tf), 1, dimnames = list(NULL, col)))
  }
  test_matrix <- as.data.frame(apply_tfidf(test_tf))
  
  test_preds <- predict(nb_model, test_matrix)
  conf_mat <- table(Predicted = test_preds, Actual = test_data$label)
  accuracy <- mean(test_preds == test_data$label)
  
  # word frequency by class (raw counts, for visuals — not TF-IDF)
  full_corpus_tf <- clean_corpus(news_data$text)
  full_dtm_tf <- DocumentTermMatrix(full_corpus_tf, control = list(weighting = weightTf))
  full_dtm_tf <- removeSparseTerms(full_dtm_tf, 0.99)
  full_matrix_tf <- as.data.frame(as.matrix(full_dtm_tf))
  colnames(full_matrix_tf) <- make.names(colnames(full_matrix_tf), unique = TRUE)
  fake_terms <- colSums(full_matrix_tf[news_data$label == "FAKE", , drop = FALSE])
  real_terms <- colSums(full_matrix_tf[news_data$label == "REAL", , drop = FALSE])
  
  list(
    model = nb_model,
    vocab = vocab,
    idf = idf,
    accuracy = accuracy,
    conf_mat = conf_mat,
    n_train = nrow(train_data),
    n_test = nrow(test_data),
    fake_terms = fake_terms,
    real_terms = real_terms
  )
}

if (file.exists(cache_file)) {
  pipeline <- readRDS(cache_file)
  # Older cached models (pre-fix) won't have $idf — force a retrain if so.
  if (is.null(pipeline$idf)) {
    pipeline <- train_pipeline()
    saveRDS(pipeline, cache_file)
  }
} else {
  pipeline <- train_pipeline()
  saveRDS(pipeline, cache_file)
}

nb_model   <- pipeline$model
vocab      <- pipeline$vocab
idf        <- pipeline$idf
fake_terms <- pipeline$fake_terms
real_terms <- pipeline$real_terms

# ---------------------------------------------------------
# 4. PREDICTION HELPER
# ---------------------------------------------------------
predict_news <- function(input_text) {
  corpus <- clean_corpus(input_text)
  dtm <- DocumentTermMatrix(corpus, control = list(dictionary = vocab, weighting = weightTf))
  tf_vec <- as.matrix(dtm)
  colnames(tf_vec) <- make.names(colnames(tf_vec), unique = TRUE)
  missing_cols <- setdiff(vocab, colnames(tf_vec))
  for (col in missing_cols) {
    tf_vec <- cbind(tf_vec, matrix(0, nrow(tf_vec), 1, dimnames = list(NULL, col)))
  }
  tf_vec <- tf_vec[, vocab, drop = FALSE]
  
  # Apply the SAME idf weights learned during training (not recomputed here)
  m <- as.data.frame(sweep(tf_vec, 2, idf[vocab], `*`))
  
  pred_class <- predict(nb_model, m, type = "class")
  pred_prob <- predict(nb_model, m, type = "raw")
  
  list(
    class = as.character(pred_class),
    prob_fake = round(pred_prob[1, "FAKE"] * 100, 1),
    prob_real = round(pred_prob[1, "REAL"] * 100, 1),
    matched_terms = colnames(tf_vec)[as.numeric(tf_vec[1, ]) > 0]
  )
}

# ---------------------------------------------------------
# 5. UI — ENHANCED with BIG GAUGES
# ---------------------------------------------------------
ui <- fluidPage(
  theme = shinytheme("flatly"),
  tags$head(
    tags$style(HTML("
      body { background-color: #fafafa; font-family: 'Segoe UI', Roboto, sans-serif; }

      /* ---- Animated Logo Container ---- */
      .logo-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 30px 0 10px 0;
        animation: float 3s ease-in-out infinite;
      }
      .logo-svg {
        width: 220px;
        height: auto;
        filter: drop-shadow(0 8px 20px rgba(255, 138, 43, 0.25));
        transition: transform 0.3s ease;
      }
      .logo-svg:hover {
        transform: scale(1.04);
      }
      .logo-sub {
        font-size: 15px;
        letter-spacing: 6px;
        font-weight: 500;
        color: #ff8a2b;
        margin-top: -2px;
        background: rgba(255, 138, 43, 0.08);
        padding: 4px 20px;
        border-radius: 40px;
        animation: pulse-sub 2.5s ease-in-out infinite;
      }

      @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
        100% { transform: translateY(0px); }
      }
      @keyframes pulse-sub {
        0% { opacity: 0.7; letter-spacing: 6px; }
        50% { opacity: 1; letter-spacing: 8px; }
        100% { opacity: 0.7; letter-spacing: 6px; }
      }

      /* ---- Navbar tweaks ---- */
      .navbar {
        background-color: #ffffff !important;
        border-bottom: 1px solid #f0f0f0;
        box-shadow: 0 1px 6px rgba(0,0,0,0.02);
      }
      .navbar-nav > li > a {
        font-weight: 500;
        color: #333 !important;
        transition: color 0.2s;
      }
      .navbar-nav > li > a:hover {
        color: #ff8a2b !important;
      }

      /* ---- Cards & panels ---- */
      .panel-card {
        background: #fff;
        border-radius: 18px;
        padding: 24px 26px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.04);
        margin-bottom: 24px;
        border: 1px solid #f4f4f4;
        transition: box-shadow 0.25s ease;
      }
      .panel-card:hover {
        box-shadow: 0 8px 32px rgba(0,0,0,0.07);
      }

      .verdict-real {
        color: #1e9e5a;
        font-weight: 800;
        font-size: 32px;
      }
      .verdict-fake {
        color: #e8442f;
        font-weight: 800;
        font-size: 32px;
      }

      .btn-analyze {
        background: linear-gradient(135deg, #ffd34d, #ff8a2b);
        border: none;
        color: #1a1a1a;
        font-weight: 700;
        padding: 12px 34px;
        border-radius: 50px;
        font-size: 16px;
        transition: transform 0.15s, box-shadow 0.2s;
        box-shadow: 0 4px 12px rgba(255, 138, 43, 0.25);
      }
      .btn-analyze:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(255, 138, 43, 0.35);
        color: #1a1a1a;
      }

      .small-note {
        color: #999;
        font-size: 12px;
        line-height: 1.5;
      }
      .footer-note {
        text-align: center;
        color: #bbb;
        font-size: 12px;
        padding: 30px 0 16px;
        border-top: 1px solid #eee;
        margin-top: 20px;
      }
      .stat-box {
        text-align: center;
        padding: 12px 8px;
        background: #fcfcfc;
        border-radius: 14px;
        border: 1px solid #f0f0f0;
      }
      .stat-num {
        font-size: 32px;
        font-weight: 800;
        color: #ff8a2b;
        line-height: 1.2;
      }
      .stat-label {
        color: #888;
        font-size: 13px;
        font-weight: 500;
      }

      /* ---- Inputs ---- */
      .form-control {
        border-radius: 12px;
        border: 1px solid #e6e6e6;
        padding: 10px 14px;
        transition: border-color 0.2s, box-shadow 0.2s;
      }
      .form-control:focus {
        border-color: #ff8a2b;
        box-shadow: 0 0 0 3px rgba(255, 138, 43, 0.15);
      }      }
=
      textarea.form-control {
        min-height: 120px;
      }

      /* ---- Tabs ---- */
      .nav-tabs > li > a {
        border-radius: 30px 30px 0 0;
        font-weight: 500;
      /* ---- BIG GAUGE STYLES ---- */
      .gauge-container {
        padding: 5px 0;
        height: 400px;
        width: 100%;
      }
      .gauge-wrapper {
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .gauge-card {
        background: #fff;
        border-radius: 18px;
        padding: 15px 20px 20px 20px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.04);
        margin-bottom: 24px;
        border: 1px solid #f4f4f4;
        transition: box-shadow 0.25s ease;
        height: 420px;
      }
      .gauge-card:hover {
        box-shadow: 0 8px 32px rgba(0,0,0,0.07);
      }
      .gauge-title {
        text-align: center;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 2px solid #f5f5f5;
      }
      .gauge-title-fake {
        color: #e8442f;
      }
      .gauge-title-real {
        color: #1e9e5a;
      }

      /* ---- responsive tweaks ---- */
      @media (max-width: 768px) {
        .logo-svg { width: 160px; }
        .logo-sub { font-size: 12px; letter-spacing: 4px; }
        .verdict-real, .verdict-fake { font-size: 24px; }
        .btn-analyze { padding: 10px 24px; font-size: 14px; }
        .gauge-container { height: 350px; }
        .gauge-card { height: 370px; }
      }
    "))
  ),
  
  # ---- BIG ANIMATED LOGO (SVG) ----
  div(class = "logo-container",
      tags$svg(
        class = "logo-svg", viewBox = "0 0 400 120", fill = "none",
        xmlns = "http://www.w3.org/2000/svg",
        `aria-label` = "ApexReality Logo",
        tags$g(
          # ---- "APEX" ----
          tags$text(x = "0", y = "80", 
                    style = "font-family: 'Segoe UI', Helvetica, sans-serif; font-weight: 800; font-size: 68px; fill: #1a1a1a; letter-spacing: 4px;",
                    "APEX"),
          # ---- "REALITY" ----
          tags$text(x = "148", y = "80", 
                    style = "font-family: 'Segoe UI', Helvetica, sans-serif; font-weight: 400; font-size: 68px; fill: #ff8a2b; letter-spacing: 2px;",
                    "REALITY"),
          # ---- decorative underline ----
          tags$line(x1 = "0", y1 = "92", x2 = "380", y2 = "92", 
                    stroke = "#ff8a2b", stroke_width = "3", stroke_linecap = "round",
                    stroke_dasharray = "6 6", opacity = "0.4"),
          # ---- small diamond accent ----
          tags$polygon(points = "390,92 396,86 402,92 396,98", fill = "#ff8a2b", opacity = "0.8")
        )
      ),
      div(class = "logo-sub", "⚡ FAKE NEWS DETECTION ⚡")
  ),
  
  # ---- TABS ----
  tabsetPanel(
    id = "main_tabs",
    # ---- ANALYZE TAB ----
    tabPanel("🔍 Analyze",
             fluidRow(
               column(5,
                      div(class = "panel-card",
                          h4("📰 Check a news article"),
                          p(class = "small-note", "Paste a headline and/or article body below. We'll analyze it in real-time."),
                          textInput("headline", "Headline (optional)", placeholder = "e.g. Scientists confirm..."),
                          textAreaInput("article_text", "Article text", rows = 10,
                                        placeholder = "Paste the news article text here..."),
                          fluidRow(
                            column(6, selectInput("source_type", "Source type",
                                                  choices = c("Unknown", "News website", "Social media post",
                                                              "Blog", "Press release", "Forwarded message"))),
                            column(6, selectInput("category", "Topic category",
                                                  choices = c("General", "Politics", "Health", "Science",
                                                              "Business", "Entertainment", "World")))
                          ),
                          br(),
                          actionButton("analyze_btn", "Analyze Article", class = "btn-analyze", icon = icon("search")),
                          br(), br(),
                          p(class = "small-note",
                            paste0("🧠 Model: Naive Bayes (TF-IDF) trained on ", nrow(news_data),
                                   " labeled articles. See 'Model Performance' tab for accuracy."))
                      )
               ),
               column(7,
                      uiOutput("result_panel")
               )
             )
    ),
    
    # ---- VISUAL INSIGHTS TAB ----
    tabPanel("📊 Visual Insights",
             fluidRow(
               column(6,
                      div(class = "gauge-card",
                          div(class = "gauge-title gauge-title-fake", "🎯 FAKE Confidence"),
                          div(class = "gauge-container",
                              div(class = "gauge-wrapper", plotlyOutput("conf_gauge_fake", height = "100%", width = "100%"))
                          )
                      )
               ),
               column(6,
                      div(class = "gauge-card",
                          div(class = "gauge-title gauge-title-real", "🎯 REAL Confidence"),
                          div(class = "gauge-container",
                              div(class = "gauge-wrapper", plotlyOutput("conf_gauge_real", height = "100%", width = "100%"))
                          )
                      )
               )
             ),
             fluidRow(
               column(12,
                      div(class = "panel-card",
                          h4("🔑 Matched Keywords in Your Text"),
                          plotlyOutput("matched_terms_plot", height = "260px")
                      )
               )
             ),
             fluidRow(
               column(6,
                      div(class = "panel-card",
                          h4("☁️ Top Words — FAKE corpus"),
                          plotOutput("wordcloud_fake", height = "280px", click = "wordcloud_fake_click"),
                          verbatimTextOutput("fake_word_info")
                      )
               ),
               column(6,
                      div(class = "panel-card",
                          h4("☁️ Top Words — REAL corpus"),
                          plotOutput("wordcloud_real", height = "280px", click = "wordcloud_real_click"),
                          verbatimTextOutput("real_word_info")
                      )
               )
             ),
             fluidRow(
               column(12,
                      div(class = "panel-card",
                          h4("📈 Most Frequent Words by Class (Training Data)"),
                          plotlyOutput("freq_compare_plot", height = "380px")
                      )
               )
             ),
             fluidRow(
               column(12,
                      div(class = "panel-card",
                          h4("🔍 Explore Word Frequencies"),
                          fluidRow(
                            column(6, selectInput("word_class_filter", "Select Word Class",
                                                  choices = c("All", "FAKE", "REAL"))),
                            column(6, selectInput("top_n_words", "Number of top words to display",
                                                  choices = c(10, 15, 20, 30, 50), selected = 15))
                          ),
                          plotlyOutput("interactive_word_freq", height = "400px")
                      )
               )
             ),
             fluidRow(
               column(12,
                      div(class = "panel-card",
                          h4("📊 Distribution Analysis"),
                          plotlyOutput("class_distribution", height = "300px")
                      )
               )
             )
    ),
    
    # ---- MODEL PERFORMANCE TAB ----
    tabPanel("📈 Model Performance",
             fluidRow(
               column(4,
                      div(class = "panel-card stat-box",
                          div(class = "stat-num", paste0(round(pipeline$accuracy * 100, 1), "%")),
                          div(class = "stat-label", "Test-set accuracy"))
               ),
               column(4,
                      div(class = "panel-card stat-box",
                          div(class = "stat-num", pipeline$n_train),
                          div(class = "stat-label", "Training articles"))
               ),
               column(4,
                      div(class = "panel-card stat-box",
                          div(class = "stat-num", pipeline$n_test),
                          div(class = "stat-label", "Held-out test articles"))
               )
             ),
             fluidRow(
               column(6,
                      div(class = "panel-card",
                          h4("📋 Confusion Matrix (test set)"),
                          tableOutput("conf_matrix_table"),
                          p(class = "small-note",
                            "Rows = predicted class, columns = actual class. Diagonal = correct predictions.")
                      )
               ),
               column(6,
                      div(class = "panel-card",
                          h4("📌 Notes"),
                          p("This model is evaluated on data held out during training (never seen by the model)."),
                          p("If accuracy is lower than expected, the usual fixes are: more training rows, ",
                            "removing near-duplicate articles, balancing classes, or adding bigrams/trigrams ",
                            "to capture phrasing patterns instead of single words."),
                          p(class = "small-note",
                            "🔄 To retrain after updating data/news_dataset.csv, delete model_cache.rds and restart the app.")
                      )
               )
             )
    ),
    
    # ---- DATASET TAB ----
    tabPanel("📚 Dataset",
             div(class = "panel-card",
                 h4("📄 Training dataset preview"),
                 p(class = "small-note", paste0(nrow(news_data), " labeled articles currently loaded.")),
                 DTOutput("dataset_table")
             )
    )
  ),
  
  div(class = "footer-note", "ApexReality © 2026 — Built with R Shiny · Trust, but verify.")
)

# ---------------------------------------------------------
# 6. SERVER — Enhanced with BIG interactive gauges
# ---------------------------------------------------------
server <- function(input, output, session) {
  
  result <- eventReactive(input$analyze_btn, {
    full_text <- paste(input$headline, input$article_text, sep = ". ")
    validate(need(nchar(trimws(input$article_text)) > 0,
                  "Please paste some article text before analyzing."))
    predict_news(full_text)
  })
  
  output$result_panel <- renderUI({
    if (input$analyze_btn == 0) {
      return(
        div(class = "panel-card",
            h4("📌 Result"),
            p("Your analysis will appear here once you submit an article.")
        )
      )
    }
    res <- result()
    verdict_class <- if (res$class == "FAKE") "verdict-fake" else "verdict-real"
    icon_name <- if (res$class == "FAKE") "triangle-exclamation" else "circle-check"
    
    tagList(
      div(class = "panel-card",
          h4("📌 Result"),
          div(icon(icon_name), span(class = verdict_class, paste(" Likely", res$class))),
          br(),
          fluidRow(
            column(6,
                   h5("Confidence: FAKE"),
                   div(style = "background:#fde2de; border-radius:8px; height:16px;",
                       div(style = paste0("background:#e8442f; width:", res$prob_fake,
                                          "%; height:16px; border-radius:8px;"))),
                   p(paste0(res$prob_fake, "%"))
            ),
            column(6,
                   h5("Confidence: REAL"),
                   div(style = "background:#d9f2e2; border-radius:8px; height:16px;",
                       div(style = paste0("background:#1e9e5a; width:", res$prob_real,
                                          "%; height:16px; border-radius:8px;"))),
                   p(paste0(res$prob_real, "%"))
            )
          ),
          tags$hr(),
          h5("Why this verdict?"),
          if (length(res$matched_terms) > 0) {
            p("Keywords in your text that matched the training vocabulary: ",
              tags$b(paste(res$matched_terms, collapse = ", ")))
          } else {
            p("No strong matching keywords found in the training vocabulary; verdict is based on overall text patterns.")
          },
          p(class = "small-note", "See the 'Visual Insights' tab for charts on this result.")
      )
    )
  })
  
  # ---- BIG GAUGE FOR FAKE CONFIDENCE ----
  output$conf_gauge_fake <- renderPlotly({
    if (input$analyze_btn == 0) {
      return(plot_ly(
        type = "indicator",
        mode = "gauge+number+delta",
        value = 0,
        number = list(suffix = "%", font = list(size = 40, color = "#999", weight = "bold")),
        delta = list(reference = 50, relative = FALSE, font = list(size = 18)),
        title = list(text = "Fake News Probability", font = list(size = 20, color = "#666")),
        gauge = list(
          axis = list(range = list(0, 100), 
                      tickvals = c(0, 25, 50, 75, 100),
                      ticktext = c("0%", "25%", "50%", "75%", "100%"),
                      tickfont = list(size = 14, color = "#666")),
          bar = list(color = "#ccc", thickness = 0.5),
          bgcolor = "white",
          borderwidth = 2,
          bordercolor = "#e0e0e0",
          steps = list(
            list(range = c(0, 33), color = "rgba(30, 158, 90, 0.15)"),
            list(range = c(33, 66), color = "rgba(255, 193, 7, 0.15)"),
            list(range = c(66, 100), color = "rgba(232, 68, 47, 0.15)")
          )
        ),
        domain = list(x = c(0.05, 0.95), y = c(0.1, 0.95))
      ) %>% 
        layout(
          margin = list(l = 30, r = 30, t = 40, b = 30),
          height = 380,
          autosize = TRUE,
          paper_bgcolor = "rgba(255,255,255,0)",
          plot_bgcolor = "rgba(255,255,255,0)"
        ))
    }
    
    res <- result()
    
    # Determine color and status
    gauge_color <- if(res$prob_fake >= 70) "#e8442f" else if(res$prob_fake >= 40) "#ffc107" else "#1e9e5a"
    status_text <- if(res$prob_fake >= 70) "⚠️ HIGH RISK" else if(res$prob_fake >= 40) "⚖️ MODERATE" else "✅ LOW RISK"
    status_color <- if(res$prob_fake >= 70) "#e8442f" else if(res$prob_fake >= 40) "#ffc107" else "#1e9e5a"
    
    plot_ly(
      type = "indicator",
      mode = "gauge+number+delta",
      value = res$prob_fake,
      number = list(
        suffix = "%", 
        font = list(size = 48, color = gauge_color, weight = "bold", family = "Arial Black")
      ),
      delta = list(
        reference = 50, 
        relative = FALSE,
        font = list(size = 20, color = "#333"),
        increasing = list(color = "#e8442f"),
        decreasing = list(color = "#1e9e5a")
      ),
      title = list(
        text = paste0("Fake News Probability<br><span style='font-size:18px;color:", status_color, ";font-weight:bold;'>", 
                      status_text,
                      "</span>"),
        font = list(size = 22, color = "#555", family = "Arial")
      ),
      gauge = list(
        axis = list(
          range = list(0, 100),
          tickvals = c(0, 20, 40, 60, 80, 100),
          ticktext = c("0", "20", "40", "60", "80", "100"),
          tickfont = list(size = 16, color = "#666", family = "Arial"),
          tickwidth = 2,
          tickcolor = "#aaa"
        ),
        bar = list(
          color = gauge_color, 
          thickness = 0.6,
          line = list(color = gauge_color, width = 3)
        ),
        bgcolor = "rgba(255,255,255,0.9)",
        borderwidth = 4,
        bordercolor = "#eee",
        steps = list(
          list(range = c(0, 33), color = "rgba(30, 158, 90, 0.25)"),
          list(range = c(33, 66), color = "rgba(255, 193, 7, 0.25)"),
          list(range = c(66, 100), color = "rgba(232, 68, 47, 0.35)")
        ),
        threshold = list(
          line = list(color = "#333", width = 6, dash = "dash"),
          thickness = 0.7,
          value = res$prob_fake
        )
      ),
      domain = list(x = c(0.05, 0.95), y = c(0.12, 0.95))
    ) %>% 
      layout(
        margin = list(l = 30, r = 30, t = 50, b = 40),
        height = 380,
        autosize = TRUE,
        paper_bgcolor = "rgba(255,255,255,0)",
        plot_bgcolor = "rgba(255,255,255,0)",
        hovermode = FALSE
      ) %>%
      config(displayModeBar = FALSE)
  })
  
  # ---- BIG GAUGE FOR REAL CONFIDENCE ----
  output$conf_gauge_real <- renderPlotly({
    if (input$analyze_btn == 0) {
      return(plot_ly(
        type = "indicator",
        mode = "gauge+number+delta",
        value = 0,
        number = list(suffix = "%", font = list(size = 40, color = "#999", weight = "bold")),
        delta = list(reference = 50, relative = FALSE, font = list(size = 18)),
        title = list(text = "Real News Probability", font = list(size = 20, color = "#666")),
        gauge = list(
          axis = list(range = list(0, 100),
                      tickvals = c(0, 25, 50, 75, 100),
                      ticktext = c("0%", "25%", "50%", "75%", "100%"),
                      tickfont = list(size = 14, color = "#666")),
          bar = list(color = "#ccc", thickness = 0.5),
          bgcolor = "white",
          borderwidth = 2,
          bordercolor = "#e0e0e0",
          steps = list(
            list(range = c(0, 33), color = "rgba(232, 68, 47, 0.15)"),
            list(range = c(33, 66), color = "rgba(255, 193, 7, 0.15)"),
            list(range = c(66, 100), color = "rgba(30, 158, 90, 0.15)")
          )
        ),
        domain = list(x = c(0.05, 0.95), y = c(0.1, 0.95))
      ) %>% 
        layout(
          margin = list(l = 30, r = 30, t = 40, b = 30),
          height = 380,
          autosize = TRUE,
          paper_bgcolor = "rgba(255,255,255,0)",
          plot_bgcolor = "rgba(255,255,255,0)"
        ))
    }
    
    res <- result()
    
    # Determine color and status
    gauge_color <- if(res$prob_real >= 70) "#1e9e5a" else if(res$prob_real >= 40) "#ffc107" else "#e8442f"
    status_text <- if(res$prob_real >= 70) "✅ HIGH CONFIDENCE" else if(res$prob_real >= 40) "⚖️ MODERATE" else "⚠️ LOW CONFIDENCE"
    status_color <- if(res$prob_real >= 70) "#1e9e5a" else if(res$prob_real >= 40) "#ffc107" else "#e8442f"
    
    plot_ly(
      type = "indicator",
      mode = "gauge+number+delta",
      value = res$prob_real,
      number = list(
        suffix = "%",
        font = list(size = 48, color = gauge_color, weight = "bold", family = "Arial Black")
      ),
      delta = list(
        reference = 50,
        relative = FALSE,
        font = list(size = 20, color = "#333"),
        increasing = list(color = "#1e9e5a"),
        decreasing = list(color = "#e8442f")
      ),
      title = list(
        text = paste0("Real News Probability<br><span style='font-size:18px;color:", status_color, ";font-weight:bold;'>", 
                      status_text,
                      "</span>"),
        font = list(size = 22, color = "#555", family = "Arial")
      ),
      gauge = list(
        axis = list(
          range = list(0, 100),
          tickvals = c(0, 20, 40, 60, 80, 100),
          ticktext = c("0", "20", "40", "60", "80", "100"),
          tickfont = list(size = 16, color = "#666", family = "Arial"),
          tickwidth = 2,
          tickcolor = "#aaa"
        ),
        bar = list(
          color = gauge_color,
          thickness = 0.6,
          line = list(color = gauge_color, width = 3)
        ),
        bgcolor = "rgba(255,255,255,0.9)",
        borderwidth = 4,
        bordercolor = "#eee",
        steps = list(
          list(range = c(0, 33), color = "rgba(232, 68, 47, 0.25)"),
          list(range = c(33, 66), color = "rgba(255, 193, 7, 0.25)"),
          list(range = c(66, 100), color = "rgba(30, 158, 90, 0.35)")
        ),
        threshold = list(
          line = list(color = "#333", width = 6, dash = "dash"),
          thickness = 0.7,
          value = res$prob_real
        )
      ),
      domain = list(x = c(0.05, 0.95), y = c(0.12, 0.95))
    ) %>% 
      layout(
        margin = list(l = 30, r = 30, t = 50, b = 40),
        height = 380,
        autosize = TRUE,
        paper_bgcolor = "rgba(255,255,255,0)",
        plot_bgcolor = "rgba(255,255,255,0)",
        hovermode = FALSE
      ) %>%
      config(displayModeBar = FALSE)
  })
  
  # Interactive matched terms plot with hover information
  output$matched_terms_plot <- renderPlotly({
    if (input$analyze_btn == 0) return(NULL)
    res <- result()
    if (length(res$matched_terms) == 0) {
      return(plot_ly() %>% 
               layout(title = "No matching keywords found",
                      xaxis = list(visible = FALSE),
                      yaxis = list(visible = FALSE)))
    }
    df <- data.frame(term = res$matched_terms, count = 1, 
                     index = 1:length(res$matched_terms))
    plot_ly(
      data = df,
      x = ~reorder(term, index),
      y = ~count,
      type = "bar",
      marker = list(color = "#ff8a2b",
                    line = list(color = "#cc6b1f", width = 1.5)),
      text = ~paste("Term:", term, "<br>Present: Yes"),
      hoverinfo = "text",
      name = "Matched Terms"
    ) %>%
      layout(
        title = "",
        xaxis = list(title = "", tickangle = -45),
        yaxis = list(title = "Presence", range = c(0, 1.5)),
        margin = list(l=50, r=20, t=20, b=80),
        showlegend = FALSE,
        hovermode = "closest"
      ) %>%
      config(displayModeBar = TRUE, modeBarButtonsToRemove = c("zoomIn2d", "zoomOut2d"))
  })
  
  # Interactive word clouds with click event feedback
  output$wordcloud_fake <- renderPlot({
    set.seed(42)
    wordcloud(names(fake_terms), as.numeric(fake_terms),
              max.words = 60, scale = c(3.2, 0.6),
              colors = brewer.pal(8, "OrRd")[3:8],
              random.order = FALSE, rot.per = 0.35)
  })
  
  output$wordcloud_real <- renderPlot({
    set.seed(42)
    wordcloud(names(real_terms), as.numeric(real_terms),
              max.words = 60, scale = c(3.2, 0.6),
              colors = brewer.pal(8, "Greens")[3:8],
              random.order = FALSE, rot.per = 0.35)
  })
  
  # Word cloud click handlers
  output$fake_word_info <- renderPrint({
    req(input$wordcloud_fake_click)
    click <- input$wordcloud_fake_click
    cat("Selected word from FAKE corpus:\n")
    cat("Term:", click$text, "\n")
    if (!is.null(click$text)) {
      term_name <- names(fake_terms)[which.min(abs(seq_along(fake_terms) - click$text))]
      if (term_name %in% names(fake_terms)) {
        cat("Frequency:", fake_terms[term_name])
      }
    }
  })
  
  output$real_word_info <- renderPrint({
    req(input$wordcloud_real_click)
    click <- input$wordcloud_real_click
    cat("Selected word from REAL corpus:\n")
    cat("Term:", click$text, "\n")
    if (!is.null(click$text)) {
      term_name <- names(real_terms)[which.min(abs(seq_along(real_terms) - click$text))]
      if (term_name %in% names(real_terms)) {
        cat("Frequency:", real_terms[term_name])
      }
    }
  })
  
  # Enhanced frequency comparison plot with interactivity
  output$freq_compare_plot <- renderPlotly({
    top_n <- 15
    fake_top <- sort(fake_terms, decreasing = TRUE)[1:top_n]
    real_top <- sort(real_terms, decreasing = TRUE)[1:top_n]
    df <- rbind(
      data.frame(term = names(fake_top), count = as.numeric(fake_top), class = "FAKE"),
      data.frame(term = names(real_top), count = as.numeric(real_top), class = "REAL")
    )
    
    p <- ggplot(df, aes(x = reorder(term, count), y = count, fill = class)) +
      geom_col(position = position_dodge(width = 0.8), 
               alpha = 0.85, 
               color = "white", 
               size = 0.3) +
      coord_flip() +
      scale_fill_manual(values = c("FAKE" = "#e8442f", "REAL" = "#1e9e5a")) +
      labs(x = NULL, y = "Frequency", fill = "Class") +
      theme_minimal() +
      theme(
        legend.position = "top",
        panel.grid.minor = element_blank(),
        axis.text.y = element_text(size = 10, face = "bold"),
        axis.text.x = element_text(size = 9),
        plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
      )
    
    ggplotly(p, tooltip = c("x", "y", "fill")) %>%
      layout(
        hoverlabel = list(font = list(size = 12)),
        legend = list(orientation = "h", y = 1.1)
      ) %>%
      config(displayModeBar = TRUE)
  })
  
  # Interactive word frequency exploration
  output$interactive_word_freq <- renderPlotly({
    top_n <- as.numeric(input$top_n_words)
    word_class <- input$word_class_filter
    
    if (word_class == "All") {
      fake_top <- sort(fake_terms, decreasing = TRUE)[1:top_n]
      real_top <- sort(real_terms, decreasing = TRUE)[1:top_n]
      df <- rbind(
        data.frame(term = names(fake_top), count = as.numeric(fake_top), class = "FAKE"),
        data.frame(term = names(real_top), count = as.numeric(real_top), class = "REAL")
      )
    } else if (word_class == "FAKE") {
      fake_top <- sort(fake_terms, decreasing = TRUE)[1:top_n]
      df <- data.frame(term = names(fake_top), count = as.numeric(fake_top), class = "FAKE")
    } else {
      real_top <- sort(real_terms, decreasing = TRUE)[1:top_n]
      df <- data.frame(term = names(real_top), count = as.numeric(real_top), class = "REAL")
    }
    
    plot_ly(
      data = df,
      x = ~reorder(term, -count),
      y = ~count,
      color = ~class,
      type = "bar",
      colors = c("FAKE" = "#e8442f", "REAL" = "#1e9e5a"),
      text = ~paste("Word:", term, "<br>Frequency:", count, "<br>Class:", class),
      hoverinfo = "text",
      marker = list(line = list(color = "white", width = 1))
    ) %>%
      layout(
        title = paste("Top", top_n, "Words in", word_class, "Class"),
        xaxis = list(title = "Word", tickangle = -45),
        yaxis = list(title = "Frequency"),
        barmode = "group",
        hovermode = "closest",
        margin = list(l=60, r=40, t=60, b=100)
      ) %>%
      config(displayModeBar = TRUE)
  })
  
  # Class distribution visualization
  output$class_distribution <- renderPlotly({
    class_counts <- table(news_data$label)
    df_class <- data.frame(
      Class = names(class_counts),
      Count = as.numeric(class_counts),
      Percentage = round(as.numeric(class_counts) / nrow(news_data) * 100, 1)
    )
    
    plot_ly(
      data = df_class,
      labels = ~Class,
      values = ~Count,
      type = "pie",
      textinfo = "label+percent",
      hoverinfo = "text",
      text = ~paste("Class:", Class, "<br>Count:", Count, "<br>Percentage:", Percentage, "%"),
      marker = list(
        colors = c("FAKE" = "#e8442f", "REAL" = "#1e9e5a"),
        line = list(color = "white", width = 2)
      ),
      pull = c(0, 0.1)
    ) %>%
      layout(
        title = "Class Distribution in Training Dataset",
        legend = list(orientation = "h", y = -0.1),
        margin = list(t=50, b=50)
      ) %>%
      config(displayModeBar = TRUE)
  })
  
  # Confusion matrix output
  output$conf_matrix_table <- renderTable({
    as.data.frame.matrix(pipeline$conf_mat)
  }, rownames = TRUE)
  
  # Dataset table
  output$dataset_table <- renderDT({
    datatable(news_data, options = list(pageLength = 8), rownames = FALSE)
  })
}

shinyApp(ui, server)