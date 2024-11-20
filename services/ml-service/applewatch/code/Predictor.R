#====================== Apple Watch Data Predictor with Tidymodels ======================

# Programmed by Arastoo Bozorgi & Glenn Tanjoh
# Email: glenntanjoh@gmail.com
# Email: ab1502@mun.ca
#=======================================================================

# Clears the environment to start fresh
rm(list = ls())

#======================== Load Required Libraries ======================

# Installs and loads required libraries with 'pacman' for better dependency management
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  imputeTS, lubridate, data.table, dplyr, tidymodels, pryr, ranger, caret, doParallel, zoo
)


#======================= Set Up Parallel Backend ======================
# Detects the number of available cores and sets up a parallel backend
num_cores <- detectCores() - 1  # Leave one core free to keep system responsive
cl <- makeCluster(num_cores)
registerDoParallel(cl)

#======================= Set Time Zone & Arguments =====================

# Sets timezone for all operations and checks argument requirements
Sys.setenv(TZ = "America/St_Johns")

args <- c(
  "C:/Users/glenn/Desktop/Walkabilly/BeapEngine2024/services/",
  "C:/Users/glenn/Desktop/Walkabilly/BeapEngine2024/services/ml-service/SavedModels/",
  "C:/Users/glenn/Desktop/Walkabilly/BeapEngine2024/services/ml-service/aggregated_fitbit_applewatch_jaeger.csv",
  "C:/Users/glenn/Desktop/Walkabilly/BeapEngine2024/services/ml-service/applewatch/data/applewatch_data.csv",
  "randomForest"
)

# Verifies that required arguments are provided
if (length(args) != 5) {
  stop("Five arguments must be supplied: main path, model path, training file, data file, model type.", call. = FALSE)
}

main_path <- args[1]
model_path <- args[2]
training_file <- args[3]
file_name <- args[4]
model <- args[5]

#===================== Utility Function for Correlation ===========================

# Computes Pearson correlation, returning NA if there's zero variance in either column
correlation <- function(x) {
  if (sd(x[, 1]) == 0 || sd(x[, 2]) == 0) {
    return(NA)  # Return NA if standard deviation is zero to avoid divide-by-zero error
  } else {
    return(cor(x[, 1], x[, 2], method = "pearson"))
  }
}

#===================== Load and Prepare Data ==========================

# Check if the file exists before loading
if (!file.exists(file_name)) {
  stop(paste("Error: File not found at specified path:", file_name))
} else {
  applewatch_data <- fread(file_name)
  message("Data loaded successfully.")
}

# Make a copy of the data.table to avoid internal self-reference issues
applewatch_data <- data.table::copy(applewatch_data)

# Define required columns
required_columns <- c("Heart", "Calories", "Steps", "Distance")

# Check if all required columns are present in the data
missing_columns <- setdiff(required_columns, names(applewatch_data))
if (length(missing_columns) > 0) {
  stop(paste("Error: The following required columns are missing from the dataset:", 
             paste(missing_columns, collapse = ", ")))
} else {
  message("All required columns are present.")
}

# Check if required columns are numeric
non_numeric_cols <- required_columns[sapply(applewatch_data[, ..required_columns], function(col) !is.numeric(col))]
if (length(non_numeric_cols) > 0) {
  stop(paste("Error: The following columns are not numeric:", paste(non_numeric_cols, collapse = ", ")))
} else {
  message("All required columns are numeric.")
}


# Applies interpolation directly to specified columns with imputeTS in-place
cat("Applying linear interpolation on specified columns...\n")
applewatch_data[, Heart := imputeTS::na_interpolation(Heart, option = "linear")]
if ("Calories" %in% names(applewatch_data)) {
  applewatch_data[, Calories := imputeTS::na_interpolation(Calories, option = "linear")]
}
applewatch_data[, Steps := imputeTS::na_interpolation(Steps, option = "linear")]
applewatch_data[, Distance := imputeTS::na_interpolation(Distance, option = "linear")]

# Check for NAs in key columns after interpolation
na_columns <- required_columns[sapply(applewatch_data[, ..required_columns], function(col) any(is.na(col)))]
if (length(na_columns) > 0) {
  stop(paste("Error: NA values found in columns after interpolation:", paste(na_columns, collapse = ", ")))
} else {
  message("No NA values in required columns.")
}


#====================== Feature Engineering ============================
# Efficient calculation of entropy, resting heart rate, normalized heart rate, and others in-place

# Use := for each column to avoid .internal.selfref issues in data.table
applewatch_data[, EntropyApplewatchHeartPerDay_LE := -sum(prop.table(table(Heart)) * log2(prop.table(table(Heart))))]
applewatch_data[, EntropyApplewatchStepsPerDay_LE := -sum(prop.table(table(Steps)) * log2(prop.table(table(Steps))))]

# Compute resting heart rate using quantile
resting_hr <- quantile(applewatch_data$Heart, 0.05, na.rm = TRUE)
applewatch_data[, RestingApplewatchHeartrate_LE := resting_hr]
applewatch_data[, NormalizedApplewatchHeartrate_LE := Heart - resting_hr]
applewatch_data[, ApplewatchIntensity_LE := NormalizedApplewatchHeartrate_LE / (220 - 40 - resting_hr)]

# Computes rolling standard deviation and other engineered features
applewatch_data[, SDNormalizedApplewatchHR_LE := zoo::rollapply(NormalizedApplewatchHeartrate_LE, width = 10, FUN = sd, fill = NA, align = "right")]
applewatch_data[, ApplewatchStepsXDistance_LE := Steps * Distance]
applewatch_data[, Heart_RollMean := zoo::rollmean(Heart, k = 5, fill = NA, align = "right")]
applewatch_data[, Steps_RollMean := zoo::rollmean(Steps, k = 5, fill = NA, align = "right")]
applewatch_data[, Heart_Lag1 := shift(Heart, type = "lag")]
applewatch_data[, Steps_Lag1 := shift(Steps, type = "lag")]
applewatch_data[, Heart_RateOfChange := c(NA, diff(Heart))]


#==================== Activity Level Labeling ==========================
# Creates activity_trimmed column based on Steps thresholds
applewatch_data[, activity_trimmed := fifelse(Steps < 60, "Sedentary",
                                              fifelse(Steps >= 60 & Steps < 100, "Moderate",
                                                      fifelse(Steps >= 100, "Vigorous", NA_character_)))]

# Drops rows with NA in activity_trimmed
applewatch_data <- applewatch_data[!is.na(activity_trimmed)]
applewatch_data[, activity_trimmed := as.factor(activity_trimmed)]

if ("activity" %in% colnames(applewatch_data)) {
  applewatch_data <- applewatch_data[, !("activity")]
  message("'activity' column detected and removed from predictors.")
} else {
  message("'activity' column not found among predictors. Proceeding...")
}

# Defines a Tidymodels recipe to preprocess data before model training
applewatch_recipe <- recipe(activity_trimmed ~ ., data = applewatch_data) %>%
  step_zv(all_nominal_predictors()) %>%  # Remove zero-variance columns
  step_naomit(all_predictors()) %>%      # Remove rows with NAs in predictors
  step_normalize(all_numeric_predictors())  # Normalize numeric predictors

# Prepares the recipe for processing and applies to data
prepared_recipe <- prep(applewatch_recipe, training = applewatch_data, verbose = TRUE)
applewatch_data_prepared <- bake(prepared_recipe, new_data = NULL)

#===================== Model and Tuning Setup ==========================
set.seed(123)
data_split <- initial_split(applewatch_data_prepared, prop = 0.8)
train_data <- training(data_split)
test_data <- testing(data_split)

rf_model <- rand_forest(
  mode = "classification",
  mtry = tune(),
  min_n = tune(),
  trees = 200
) %>%
  set_engine("ranger", num.threads = 15)

rf_workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(applewatch_recipe)

#===================== Hyperparameter Tuning ============================
set.seed(123)
applewatch_folds <- vfold_cv(train_data, v = 5)


rf_result <- rf_workflow %>%
  tune_grid(
    resamples = applewatch_folds,
    grid = 20,
    control = control_grid(save_pred = TRUE, verbose = TRUE),
    metrics = metric_set(roc_auc, accuracy, spec)
  )

autoplot(rf_result)

predictions <- rf_result %>% collect_predictions()

rf_best <- rf_result %>% select_best(metric = "accuracy")

#===================== Finalize Model ================================
final_rf_model <- finalize_model(rf_model, rf_best)

final_rf_workflow <- workflow() %>%
  add_model(final_rf_model) %>%
  add_recipe(applewatch_recipe)

#================= Training and Evaluation on Test Data ===============
final_rf_fit <- final_rf_workflow %>% fit(data = train_data)


#================= Measure Execution Time and Memory ===================

# Measures and prints execution time and memory usage for predictions
start_time <- Sys.time()
start_mem <- pryr::mem_used()

# Predictions and evaluation on test data
test_predictions <- predict(final_rf_fit, test_data, type = "class") %>%
  bind_cols(test_data)

end_time <- Sys.time()
end_mem <- pryr::mem_used()

# Calculate accuracy
accuracy <- mean(test_predictions$.pred_class == test_predictions$activity_trimmed)
print(paste("Final model accuracy on test data:", accuracy))

# Print execution time and memory usage for predictions
execution_time <- end_time - start_time
memory_used <- end_mem - start_mem
print(paste("Execution time for predictions:", execution_time))
print(paste("Memory used for predictions:", memory_used, "bytes"))

#======================= Save Model and Predictions ============================
output_file <- paste0(main_path, "ml-service/applewatch/data/output/applewatch_data_predicted_TinyModels.csv")

# Check if the directory for saving output exists
if (!dir.exists(dirname(output_file))) {
  stop(paste("Error: The specified directory does not exist:", dirname(output_file)))
} else {
  # If the directory exists, proceed with saving the file
  write.csv(predictions, output_file, row.names = FALSE)
  message("Predictions saved successfully.")
}



# Save the fitted model using saveRDS
model_file <- paste0(model_path, "Tidymodels_RFModel_AppleWatch.rds")
saveRDS(final_rf_fit, file = model_file)

# Stop the cluster after training is complete
stopCluster(cl)
registerDoSEQ()  # Return to sequential processing
print("Model training and prediction process completed successfully!")
