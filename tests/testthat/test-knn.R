test_that("classification works vs class::knn on iris subset", {
  skip_if_not_installed("class")
  set.seed(123)
  idx <- sample(nrow(iris), 100)
  xtr <- as.matrix(iris[idx, 1:4])
  ytr <- iris[idx, 5]
  xte <- as.matrix(iris[-idx, 1:4])
  yte <- iris[-idx, 5]

  m <- knn_fit(xtr, ytr, task = "classification")
  p1 <- knn_predict(m, xte, k = 5, weights = "uniform")

  p_ref <- class::knn(train = xtr, test = xte, cl = ytr, k = 5)
  acc1 <- mean(p1 == yte)
  acc2 <- mean(p_ref == yte)

  expect_equal(length(p1), length(p_ref))
  expect_gt(acc1, 0.80)
  expect_gt(acc2, 0.80)
})

test_that("regression works vs FNN::knn.reg on mtcars", {
  skip_if_not_installed("FNN")
  set.seed(1)
  x <- as.matrix(mtcars[, -1])
  y <- mtcars$mpg
  idx <- sample(nrow(x), 20)
  m <- knn_fit(x[-idx, ], y[-idx], task = "regression")
  pr <- knn_predict(m, x[idx, ], k = 3)

  ref <- FNN::knn.reg(train = x[-idx, ], test = x[idx, ], y = y[-idx], k = 3)$pred
  expect_equal(length(pr), length(ref))
  expect_lt(mean(abs(pr - ref)), 5)
})
