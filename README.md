# knnpack

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Correctness](https://img.shields.io/badge/correctness-verified%20against%20class::knn-brightgreen)
![Regression Verified](https://img.shields.io/badge/regression-verified%20against%20FNN::knn.reg-brightgreen)
![Vignette](https://img.shields.io/badge/vignette-available-blue)
---

`knnpack` is a lightweight R package that implements
**k-Nearest Neighbors (KNN)** for classification and regression, with optional
Rcpp distance acceleration. Built for educational purposes.

---

## Installation

```r
# install.packages("devtools")
devtools::install_github("NamanDhariwal/knnpack")
```

---

## Classification Example

```r
library(knnpack)

x <- as.matrix(iris[, 1:4])
y <- iris$Species

set.seed(1)
idx <- sample(nrow(x), 100)

m <- knn_fit(x[idx, ], y[idx], task = "classification")
pred <- knn_predict(m, x[-idx, ], k = 5)

mean(pred == y[-idx])  # accuracy
```

---

## Regression Example

```r
x <- as.matrix(mtcars[, -1])
y <- mtcars$mpg

set.seed(2)
idx <- sample(nrow(x), 20)

mreg <- knn_fit(x[-idx, ], y[-idx], task = "regression")
pred <- knn_predict(mreg, x[idx, ], k = 3)

sqrt(mean((pred - y[idx])^2))  # RMSE
```

---

## Vignette

For a full walkthrough:

```r
browseVignettes("knnpack")
```
