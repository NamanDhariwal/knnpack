#==============================
# knn_fit(): Fit a KNN model
#==============================
# This function prepares the training data for KNN by:
#  - validating inputs
#  - determining the task type (classification/regression)
#  - standardizing features (optional)
#  - storing training data and metadata in a structured model object
knn_fit <- function(x, y, task = c("classification","regression"), standardize = TRUE) {

  # Validate training inputs and coerce to proper formats
  ch <- .check_train(x, y)
  x <- ch$x; y <- ch$y

  # Infer task automatically if not provided:
  #  - numeric y → regression
  #  - non-numeric y → classification
  if (missing(task)) {
    task <- if (is.numeric(y)) "regression" else "classification"
  } else {
    task <- match.arg(task)
  }

  # Convert numeric labels to factors for classification problems
  if (task == "classification" && is.numeric(y)) y <- factor(y)

  # Standardize predictors (mean = 0, sd = 1) unless disabled
  if (standardize) {
    st <- .standardize(x)
    xz <- st$x
  } else {
    st <- list(center = rep(0, ncol(x)), scale = rep(1, ncol(x)))
    xz <- x
  }

  # Return a structured model object containing:
  #  - standardized training data
  #  - labels
  #  - original standardization parameters
  #  - task type
  structure(list(
    x_train = xz,
    y_train = y,
    center = st$center,
    scale = st$scale,
    task = task
  ), class = "knnpack_model")
}



#==============================================
# knn_predict(): Predict labels/values with KNN
#==============================================
# This function computes distances between new observations and
# training points, selects the k nearest neighbors, and performs:
#  - majority vote for classification
#  - weighted average for regression
knn_predict <- function(model, newx, k = 5,
                        weights = c("uniform","distance"),
                        method = c("auto","brute"),
                        use_rcpp = TRUE) {

  # Ensure the model was created by knn_fit()
  stopifnot(inherits(model, "knnpack_model"))

  weights <- match.arg(weights)
  method  <- match.arg(method)

  # Convert data frame to matrix if needed
  if (is.data.frame(newx)) newx <- as.matrix(newx)
  if (!is.matrix(newx) || !is.numeric(newx))
    stop("newx must be numeric matrix/data.frame")

  # Apply the same standardization used on the training data
  newz <- .apply_standardization(newx, model$center, model$scale)

  #-------------------------------
  # Distance computation
  #-------------------------------
  # Prefer fast Rcpp-based distance function when available
  D <- if (use_rcpp) {
    if (!requireNamespace("Rcpp", quietly = TRUE)) {
      use_rcpp <- FALSE
    }
    if (use_rcpp) {
      # Fast C++ distance computation
      euclidean_distance(newz, model$x_train)
    } else {
      # Pure R fallback: slow but reliable
      n_new <- nrow(newz)
      n_tr  <- nrow(model$x_train)
      Dtmp  <- matrix(0, n_new, n_tr)
      for (i in seq_len(n_new)) {
        di <- sqrt(rowSums((t(t(model$x_train) - newz[i, ]))^2))
        Dtmp[i, ] <- di
      }
      Dtmp
    }
  } else {
    # Pure R method always
    n_new <- nrow(newz); n_tr <- nrow(model$x_train)
    Dtmp <- matrix(0, n_new, n_tr)
    for (i in seq_len(n_new)) {
      di <- sqrt(rowSums((t(t(model$x_train) - newz[i, ]))^2))
      Dtmp[i, ] <- di
    }
    Dtmp
  }

  # Prepare output container
  n_new <- nrow(D)
  preds <- vector(
    mode = if (model$task == "regression") "numeric" else "list",
    length = n_new
  )

  #-------------------------------
  # Loop over new observations
  #-------------------------------
  for (i in seq_len(n_new)) {

    # Identify k nearest neighbors by sorting distances
    ord <- order(D[i, ], decreasing = FALSE)
    idx <- ord[seq_len(min(k, length(ord)))]

    neigh_y <- model$y_train[idx]   # neighbor labels/values
    neigh_d <- D[i, idx]            # neighbor distances

    # Weighting scheme:
    #  - "uniform": all neighbors count equally
    #  - "distance": closer neighbors get larger weight
    if (weights == "distance") {
      w <- 1 / pmax(neigh_d, .Machine$double.eps)
    } else {
      w <- rep(1, length(idx))
    }

    #-------------------------------
    # Classification prediction
    #-------------------------------
    if (model$task == "classification") {

      # Weighted vote: sum weights for each class
      labs <- as.character(neigh_y)
      tall <- tapply(w, labs, sum)

      # Pick class with highest total weight
      pred <- names(tall)[which.max(tall)]

      preds[[i]] <- type.convert(pred, as.is = TRUE)

    #-------------------------------
    # Regression prediction
    #-------------------------------
    } else {
      preds[i] <- sum(w * neigh_y) / sum(w)
    }
  }

  # Convert to appropriate output format
  if (model$task == "classification") {
    out <- unlist(preds)
    if (is.factor(model$y_train)) {
      out <- factor(out, levels = levels(model$y_train))
    }
    out
  } else {
    as.numeric(preds)
  }
}



#==============================================
# knn_score(): Evaluate KNN model performance
#==============================================
# Computes:
#  - Classification: accuracy = mean(pred == y)
#  - Regression: RMSE = sqrt(mean((pred - y)^2))
# Accepts additional arguments passed to knn_predict().
knn_score <- function(model, x, y, k = 5, ...) {

  # Generate predictions using knn_predict
  pred <- knn_predict(model, x, k = k, ...)

  # Classification accuracy
  if (model$task == "classification") {
    mean(pred == y)

  # Regression RMSE
  } else {
    sqrt(mean((pred - y)^2))
  }
}
