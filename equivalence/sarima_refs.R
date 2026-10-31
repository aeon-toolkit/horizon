# R reference forecasts for equivalence/sarima.py: stats::arima with the order
# given, method "CSS" and "CSS-ML", and on simulated series forecast::auto.arima.
#
#   Rscript equivalence/sarima_refs.R series.csv r.csv [n_workers]
#
# Coefficients are written in the order ar, ma, sar, sma, intercept.
library(parallel)

args <- commandArgs(trailingOnly = TRUE)
series <- read.csv(args[1], stringsAsFactors = FALSE)
workers <- if (length(args) > 2) as.integer(args[3]) else 8L

one <- function(i, series) {
  row <- series[i, ]
  values <- as.numeric(strsplit(row$y, ";")[[1]])
  y <- values[seq_len(length(values) - row$h)]
  out <- vector("list", 2)
  methods <- c("CSS", "CSS-ML")
  for (k in seq_along(methods)) {
    start <- Sys.time()
    res <- tryCatch({
      fit <- stats::arima(
        ts(y, frequency = row$m),
        order = c(row$p, row$d, row$q),
        seasonal = list(order = c(row$P, row$D, row$Q), period = row$m),
        include.mean = row$constant == 1,
        method = methods[k]
      )
      list(
        fc = as.numeric(predict(fit, n.ahead = row$h)$pred),
        coef = as.numeric(coef(fit)), error = NA_character_
      )
    }, error = function(e) {
      list(fc = rep(NA_real_, row$h), coef = NA_real_,
           error = substr(conditionMessage(e), 1, 200))
    })
    if (is.na(res$error) && !all(is.finite(res$fc))) res$error <- "non-finite forecast"
    out[[k]] <- data.frame(
      id = row$id, toolkit = paste("R", methods[k]), aic = NA_real_,
      time = as.numeric(difftime(Sys.time(), start, units = "secs")),
      error = res$error,
      fc = paste(format(res$fc, digits = 11, trim = TRUE), collapse = ";"),
      coef = paste(format(res$coef, digits = 12, trim = TRUE), collapse = ";"),
      stringsAsFactors = FALSE
    )
  }
  if (row$group == "simulated") {
    start <- Sys.time()
    res <- tryCatch({
      fit <- forecast::auto.arima(ts(y, frequency = row$m))
      list(fc = as.numeric(forecast::forecast(fit, h = row$h)$mean), error = NA_character_)
    }, error = function(e) {
      list(fc = rep(NA_real_, row$h), error = substr(conditionMessage(e), 1, 200))
    })
    out[[3]] <- data.frame(
      id = row$id, toolkit = "R auto.arima", aic = NA_real_,
      time = as.numeric(difftime(Sys.time(), start, units = "secs")),
      error = res$error,
      fc = paste(format(res$fc, digits = 11, trim = TRUE), collapse = ";"),
      coef = "", stringsAsFactors = FALSE
    )
  }
  do.call(rbind, out)
}

cl <- makeCluster(workers)
res <- parLapplyLB(cl, seq_len(nrow(series)), one, series = series, chunk.size = 16)
stopCluster(cl)
write.csv(do.call(rbind, res), args[2], row.names = FALSE, na = "")

# one thread, airline model, median of 5
timing <- do.call(rbind, lapply(c(100, 1000, 10000), function(n) {
  set.seed(42 + n)
  e <- rnorm(n + 13)
  w <- stats::filter(e, c(1, -0.4, rep(0, 10), -0.6, 0.24), sides = 1)
  y <- diffinv(diffinv(as.numeric(w[-(1:13)]), lag = 12))
  do.call(rbind, lapply(c("CSS", "CSS-ML"), function(method) {
    times <- sapply(1:5, function(r) {
      system.time({
        fit <- stats::arima(ts(y, frequency = 12), order = c(0, 1, 1),
                            seasonal = list(order = c(0, 1, 1), period = 12),
                            method = method)
        predict(fit, n.ahead = 12)
      })[["elapsed"]]
    })
    data.frame(toolkit = paste("R", method), n = n, median_s = median(times))
  }))
}))
write.csv(timing, file.path(dirname(args[2]), "timing_r.csv"), row.names = FALSE)
