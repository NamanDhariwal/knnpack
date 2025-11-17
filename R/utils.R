#' @keywords internal
.numlike <- function(x) is.numeric(x) && is.matrix(x)

#' @keywords internal
.check_train <- function(x, y) {
  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x)) stop("x must be a matrix or data.frame")
  if (!is.numeric(x)) stop("x must be numeric")
  if (nrow(x) != length(y)) stop("Rows of x must match length of y")
  list(x = x, y = y)
}

#' @keywords internal
.standardize <- function(x, center = TRUE, scale = TRUE) {
  cx <- x
  mu <- rep(0, ncol(x)); sc <- rep(1, ncol(x))
  if (center) mu <- colMeans(x)
  if (scale)  sc <- apply(x, 2, function(col) { s <- sd(col); if (s == 0) 1 else s })
  cx <- sweep(x, 2, mu, "-")
  cx <- sweep(cx, 2, sc, "/")
  list(x = cx, center = mu, scale = sc)
}

#' @keywords internal
.apply_standardization <- function(x, center, scale) {
  x <- sweep(x, 2, center, "-")
  x <- sweep(x, 2, scale, "/")
  x
}

#' @keywords internal
.mode <- function(v) {
  if (length(v) == 0) return(NA)
  vt <- table(v)
  nm <- names(vt)[which.max(vt)]
  type.convert(nm, as.is = TRUE)
}
