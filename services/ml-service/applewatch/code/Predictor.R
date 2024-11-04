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
  "C:/Users/glenn/Desktop/BeapR/term-project-2024-team-3/services/",
  "C:/Users/glenn/Desktop/BeapR/term-project-2024-team-3/services/ml-service/SavedModels/",
  "C:/Users/glenn/Desktop/BeapR/term-project-2024-team-3/services/ml-service/aggregated_fitbit_applewatch_jaeger.csv",
  "C:/Users/glenn/Desktop/BeapR/term-project-2024-team-3/services/ml-service/applewatch/data/applewatch_data.csv",
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

# Loads Apple Watch data, checks for required columns, adds them if missing
applewatch_data <- fread(file_name)

# Quickly adds missing columns in-place if required, filling with NA
required_columns <- c("Heart", "Calories", "Steps", "Distance")
missing_columns <- setdiff(required_columns, names(applewatch_data))
if (length(missing_columns) > 0) {
  applewatch_data[, (missing_columns) := NA]
}

# Applies interpolation directly to specified columns with imputeTS in-place
cat("Applying linear interpolation on specified columns...\n")
applewatch_data[, Heart := imputeTS::na_interpolation(Heart, option = "linear")]
if ("Calories" %in% names(applewatch_data)) {
  applewatch_data[, Calories := imputeTS::na_interpolation(Calories, option = "linear")]
}
applewatch_data[, Steps := imputeTS::na_interpolation(Steps, option = "linear")]
applewatch_data[, Distance := imputeTS::na_interpolation(Distance, option = "linear")]

# Generates new engineered features for model training
applewatch_data <- applewatch_data %>%
  mutate(
    EntropyApplewatchHeartPerDay_LE = -sum(table(Heart) / length(Heart) * log2(table(Heart) / length(Heart))),
    EntropyApplewatchStepsPerDay_LE = -sum(table(Steps) / length(Steps) * log2(table(Steps) / length(Steps))),
    RestingApplewatchHeartrate_LE = quantile(Heart, 0.05, na.rm = TRUE),
    NormalizedApplewatchHeartrate_LE = Heart - RestingApplewatchHeartrate_LE,
    ApplewatchIntensity_LE = NormalizedApplewatchHeartrate_LE / (220 - 40 - RestingApplewatchHeartrate_LE),
    SDNormalizedApplewatchHR_LE = rollapply(NormalizedApplewatchHeartrate_LE, width = 10, FUN = sd, by.column = FALSE, fill = NA),
    ApplewatchStepsXDistance_LE = Steps * Distance
  )

#====================== Feature Engineering ============================
# Efficient calculation of entropy, resting heart rate, normalized heart rate, and others in-place
applewatch_data[, EntropyApplewatchHeartPerDay_LE := -sum(prop.table(table(Heart)) * log2(prop.table(table(Heart))))] 
applewatch_data[, EntropyApplewatchStepsPerDay_LE := -sum(prop.table(table(Steps)) * log2(prop.table(table(Steps))))]

# Generate additional features: rolling means, lags, and rate of change
applewatch_data[, Heart_RollMean := zoo::rollmean(Heart, k = 5, fill = NA, align = "right")]
applewatch_data[, Steps_RollMean := zoo::rollmean(Steps, k = 5, fill = NA, align = "right")]
applewatch_data[, Heart_Lag1 := shift(Heart, type = "lag")]
applewatch_data[, Steps_Lag1 := shift(Steps, type = "lag")]
applewatch_data[, Heart_RateOfChange := c(NA, diff(Heart))]

# Computes resting heart rate using quantile, then normalizes and calculates intensity measures
resting_hr <- quantile(applewatch_data$Heart, 0.05, na.rm = TRUE)
applewatch_data[, RestingApplewatchHeartrate_LE := resting_hr]
applewatch_data[, NormalizedApplewatchHeartrate_LE := Heart - resting_hr]
applewatch_data[, ApplewatchIntensity_LE := NormalizedApplewatchHeartrate_LE / (220 - 40 - resting_hr)]


# Computes rolling standard deviation for heart rate in-place with zoo's rollapply
applewatch_data[, SDNormalizedApplewatchHR_LE := zoo::rollapply(NormalizedApplewatchHeartrate_LE, width = 10, FUN = sd, fill = NA, align = "right")]
applewatch_data[, ApplewatchStepsXDistance_LE := Steps * Distance]

#==================== Activity Level Labeling ==========================
# Creates activity_trimmed column based on Steps thresholds
applewatch_data[, activity_trimmed := fifelse(Steps < 60, "Sedentary",
                                              fifelse(Steps >= 60 & Steps < 100, "Moderate",
                                                      fifelse(Steps >= 100, "Vigorous", NA_character_)))]

# Drops rows with NA in activity_trimmed
applewatch_data <- applewatch_data[!is.na(activity_trimmed)]
applewatch_data[, activity_trimmed := as.factor(activity_trimmed)]

# Defines a Tidymodels recipe to preprocess data before model training
applewatch_recipe <- recipe(activity_trimmed ~ ., data = applewatch_data) %>%
  step_zv(all_nominal_predictors()) %>%  # Remove zero-variance columns
  step_naomit(all_predictors()) %>%      # Remove rows with NAs in predictors
  step_normalize(all_numeric_predictors())  # Normalize numeric predictors

# Prepares the recipe for processing and applies to data
prepared_recipe <- prep(applewatch_recipe, training = applewatch_data, verbose = TRUE)
applewatch_data_prepared <- bake(prepared_recipe, new_data = NULL)

#====================== Train Model with Tidymodels ====================

# Sets up a Random Forest model using Tidymodels' workflow
rf_model <- rand_forest(mode = "classification", mtry = 3, trees = 100) %>%
  set_engine("ranger", num.threads = num_cores) # Set number of threads to the number of cores

# Creates a Tidymodels workflow with the model and recipe
rf_workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(applewatch_recipe)

# Splits data into train-test sets, trains the model, and makes predictions
set.seed(123)
data_split <- initial_split(applewatch_data_prepared, prop = 0.8)
train_data <- training(data_split)
test_data <- testing(data_split)

# Fit the model on training data
rf_fit <- fit(rf_workflow, data = train_data)

#================= Measure Execution Time and Memory ===================

# Measures and prints execution time and memory usage for predictions
start_time <- Sys.time()
start_mem <- pryr::mem_used()

# Make predictions on test data
predictions <- predict(rf_fit, test_data, type = "class") %>%
  bind_cols(test_data)

end_time <- Sys.time()
end_mem <- pryr::mem_used()

# Calculate accuracy
accuracy <- mean(predictions$.pred_class == predictions$activity_trimmed)
print(paste("Accuracy of the Tidymodels model:", accuracy))

# Print execution time and memory usage for predictions
execution_time <- end_time - start_time
memory_used <- end_mem - start_mem
print(paste("Execution time for predictions:", execution_time))
print(paste("Memory used for predictions:", memory_used, "bytes"))

#======================= Save Model and Predictions ============================
output_file <- paste0(main_path, "output/applewatch_data_predicted_TinyModels.csv")
write.csv(predictions, output_file, row.names = FALSE)

# Save the fitted model using saveRDS
model_file <- paste0(model_path, "Tidymodels_RFModel_AppleWatch.rds")
saveRDS(rf_fit, file = model_file)

# Stop the cluster after training is complete
stopCluster(cl)
registerDoSEQ()  # Return to sequential processing
print("Model training and prediction process completed successfully!")
