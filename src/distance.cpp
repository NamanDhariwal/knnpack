#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

// x: n x p, y: m x p; returns n x m distance matrix (Euclidean)
// [[Rcpp::export]]
arma::mat euclidean_distance(const arma::mat& x, const arma::mat& y) {
  arma::mat out(x.n_rows, y.n_rows, arma::fill::zeros);
  for (arma::uword i = 0; i < x.n_rows; ++i) {
    for (arma::uword j = 0; j < y.n_rows; ++j) {
      out(i, j) = arma::norm(x.row(i) - y.row(j), 2);
    }
  }
  return out;
}
