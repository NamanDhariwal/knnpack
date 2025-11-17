knn_fit <- function(x, y, task = c("classification","regression"), standardize = TRUE) {
  ch <- .check_train(x, y)
  x <- ch$x; y <- ch$y

  if (missing(task)) {
    task <- if (is.numeric(y)) "regression" else "classification"
  } else {
    task <- match.arg(task)
  }
  if (task == "classification" && is.numeric(y)) y <- factor(y)

  if (standardize) {
    st <- .standardize(x)
    xz <- st$x
  } else {
    st <- list(center = rep(0, ncol(x)), scale = rep(1, ncol(x)))
    xz <- x
  }

  structure(list(
    x_train = xz,
    y_train = y,
    center = st$center,
    scale = st$scale,
    task = task
  ), class = "knnpack_model")
}

knn_predict <- function(model, newx, k = 5,
                        weights = c("uniform","distance"),
                        method = c("auto","brute"),
                        use_rcpp = TRUE) {
  stopifnot(inherits(model, "knnpack_model"))
  weights <- match.arg(weights)
  method  <- match.arg(method)

  if (is.data.frame(newx)) newx <- as.matrix(newx)
  if (!is.matrix(newx) || !is.numeric(newx)) stop("newx must be numeric matrix/data.frame")

  newz <- .apply_standardization(newx, model$center, model$scale)

  D <- if (use_rcpp) {
    if (!requireNamespace("Rcpp", quietly = TRUE)) {
      use_rcpp <- FALSE
    }
    if (use_rcpp) {
      euclidean_distance(newz, model$x_train) 
    } else {
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
    n_new <- nrow(newz); n_tr <- nrow(model$x_train)
    Dtmp <- matrix(0, n_new, n_tr)
    for (i in seq_len(n_new)) {
      di <- sqrt(rowSums((t(t(model$x_train) - newz[i, ]))^2))
      Dtmp[i, ] <- di
    }
    Dtmp
  }

  n_new <- nrow(D)
  preds <- vector(mode = if (model$task == "regression") "numeric" else "list", length = n_new)

  for (i in seq_len(n_new)) {
    ord <- order(D[i, ], decreasing = FALSE)
    idx <- ord[seq_len(min(k, length(ord)))]
    neigh_y <- model$y_train[idx]
    neigh_d <- D[i, idx]
    if (weights == "distance") {
      w <- 1 / pmax(neigh_d, .Machine$double.eps)
    } else {
      w <- rep(1, length(idx))
    }

    if (model$task == "classification") {
      labs <- as.character(neigh_y)
      tall <- tapply(w, labs, sum)
      pred <- names(tall)[which.max(tall)]
      preds[[i]] <- type.convert(pred, as.is = TRUE)
    } else {
      preds[i] <- sum(w * neigh_y) / sum(w)
    }
  }

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

knn_score <- function(model, x, y, k = 5, ...) {
  pred <- knn_predict(model, x, k = k, ...)
  if (model$task == "classification") {
    mean(pred == y)
  } else {
    sqrt(mean((pred - y)^2))
  }
}
