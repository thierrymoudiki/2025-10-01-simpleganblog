library(keras3)
library(tensorflow)
library(compiler)

# Enable JIT compilation for performance
enableJIT(3)

set.seed(123)
set_random_seed(123)

# Define GAN class once (reused for both examples)
gan <- keras3::new_model_class(
  classname = "GAN",
  initialize = function(discriminator, generator, latent_dim) {
    super$initialize()
    self$discriminator <- discriminator
    self$generator <- generator
    self$latent_dim <- latent_dim
    self$d_loss_metric <- metric_mean(name = "d_loss")
    self$g_loss_metric <- metric_mean(name = "g_loss")
  },
  compile = function(d_optimizer, g_optimizer, loss_fn) {
    super$compile()
    self$d_optimizer <- d_optimizer
    self$g_optimizer <- g_optimizer
    self$loss_fn <- loss_fn
  },
  metrics = mark_active(function() {
    list(self$d_loss_metric, self$g_loss_metric)
  }),
  train_step = function(real_data) {
    batch_size <- tf$shape(real_data)[1]
    random_latent_vectors <- tf$random$normal(shape = c(batch_size, self$latent_dim))
    generated_data <- self$generator(random_latent_vectors)
    combined_data <- tf$concat(list(generated_data, real_data), axis = 0L)
    labels <- tf$concat(list(tf$ones(tuple(batch_size, 1L)),
                             tf$zeros(tuple(batch_size, 1L))), axis = 0L)
    labels <- labels + tf$random$uniform(tf$shape(labels), maxval = 0.01)
    
    with(tf$GradientTape() %as% tape, {
      predictions <- self$discriminator(combined_data)
      d_loss <- self$loss_fn(labels, predictions)
    })
    
    grads <- tape$gradient(d_loss, self$discriminator$trainable_weights)
    self$d_optimizer$apply_gradients(
      zip_lists(grads, self$discriminator$trainable_weights)
    )
    
    random_latent_vectors <- tf$random$normal(shape = c(batch_size, self$latent_dim))
    misleading_labels <- tf$zeros(tuple(batch_size, 1L))
    
    with(tf$GradientTape() %as% tape, {
      predictions <- random_latent_vectors |>
        self$generator() |>
        self$discriminator()
      g_loss <- self$loss_fn(misleading_labels, predictions)
    })
    
    grads <- tape$gradient(g_loss, self$generator$trainable_weights)
    self$g_optimizer$apply_gradients(
      zip_lists(grads, self$generator$trainable_weights)
    )
    
    self$d_loss_metric$update_state(d_loss)
    self$g_loss_metric$update_state(g_loss)
    list(d_loss = self$d_loss_metric$result(),
         g_loss = self$g_loss_metric$result())
  }
)

# Reusable training function
train_gan <- function(train_dat, generator_fn, discriminator_fn, 
                      n_iter = 5, epochs_per_iter = 30, 
                      num_resamples = 500, plot_title = "GAN Training",
                      true_density_fn = NULL) {
  
  latent_dim <- as.integer(1)
  nsyn <- nrow(train_dat)
  
  # Build and compile model
  mod <- gan(
    discriminator = discriminator_fn(dim = ncol(train_dat)), 
    generator = generator_fn(latent_dim = latent_dim), 
    latent_dim = latent_dim
  )
  
  mod$compile(
    d_optimizer = optimizer_adam(beta_1 = 0.5),
    g_optimizer = optimizer_adam(beta_1 = 0.5),
    loss_fn = loss_binary_crossentropy()
  )
  
  # Initial plot
  plot(density(train_dat), ylim = c(0, 0.2), xlim = c(-12, 16), 
       col = "darkorange2", xlab = "x", main = plot_title)
  if (!is.null(true_density_fn)) true_density_fn()
  
  # Training loop
  start <- proc.time()[3]
  pb <- utils::txtProgressBar(max = n_iter, style = 3L)
  
  for (i in seq_len(n_iter)) {
    mod |> fit(train_dat, epochs = epochs_per_iter, batch_size = 32, verbose = 0)
    newdat <- mod$generator(tf$random$normal(shape = c(nsyn, latent_dim)))
    lines(density(as.matrix(newdat)), 
          col = RColorBrewer::brewer.pal(max(7, n_iter), "Greens")[i])
    utils::setTxtProgressBar(pb, i)
  }
  
  elapsed <- proc.time()[3] - start
  close(pb)
  cat(sprintf("\nElapsed %.2f seconds\n", elapsed))
  
  # Generate resamples
  latent_vectors <- matrix(rnorm(n = num_resamples * latent_dim), 
                           nrow = num_resamples, ncol = latent_dim)
  resamples_matrix <- as.matrix(mod$generator(latent_vectors))
  
  # Final plots
  plot(density(train_dat), col = "darkorange2", 
       xlim = c(-12, 16), ylim = c(0, 0.2), 
       xlab = "x", main = "Generated Resamples vs Real Data")
  lines(density(resamples_matrix), col = "darkgreen", lty = 2)
  
  hist(train_dat, col = rgb(1, 0.5, 0, 0.5), xlim = c(-12, 16), 
       main = "Histogram of Real vs Generated Data", 
       xlab = "x", probability = TRUE)
  hist(resamples_matrix, col = rgb(0, 0.5, 0, 0.5), add = TRUE, probability = TRUE)
  legend("topright", legend = c("Real Data", "Generated Data"), 
         fill = c(rgb(1, 0.5, 0, 0.5), rgb(0, 0.5, 0, 0.5)))
  
  return(list(model = mod, resamples = resamples_matrix, time = elapsed))
}

# ============================================================================
# 1 - UNIMODAL ----
# ============================================================================

N <- 1000
train_dat <- matrix(rnorm(n = N, mean = 2, sd = 4))

plot(density(train_dat), col = "darkorange2")
curve(dnorm(x, 2, 4), add = TRUE)

# Architecture functions
generator_unimodal <- function(latent_dim = 1) {
  keras_model_sequential(input_shape = latent_dim, name = "seq_gen") |> 
    layer_dense(units = 1, activation = "linear")
}

discriminator_unimodal <- function(dim = 1) {
  keras_model_sequential(input_shape = dim, name = "seq_disc") |> 
    layer_dense(units = 8, activation = "relu") |>
    layer_dense(units = 1, activation = "sigmoid")
}

summary(generator_unimodal())
summary(discriminator_unimodal())

# Train
result1 <- train_gan(
  train_dat = train_dat,
  generator_fn = generator_unimodal,
  discriminator_fn = discriminator_unimodal,
  n_iter = 5,
  epochs_per_iter = 30,
  num_resamples = 500,
  plot_title = "GAN Training - Unimodal",
  true_density_fn = function() {
    curve(dnorm(x, 2, 4), add = TRUE)
    curve(dnorm(x, 0, 1), add = TRUE, col = "black", lty = 2)
  }
)

# ============================================================================
# 2 - MIXTURE ----
# ============================================================================

set.seed(123)

mixture <- rbinom(N, 1, 0.5)
train_dat <- matrix(mixture * rnorm(N, 2, 1) + (1 - mixture) * rnorm(N, 8, 2))

plot(density(train_dat), ylim = c(0, 0.2), col = "darkorange2")
curve(0.5 * dnorm(x, 2, 1) + 0.5 * dnorm(x, 8, 2), add = TRUE)

# Architecture functions
generator_mixture <- function(latent_dim = 1) {
  keras_model_sequential(input_shape = latent_dim, name = "seq_gen") |> 
    layer_dense(units = 16, activation = "relu") |>
    layer_dense(units = 1, activation = "linear")
}

discriminator_mixture <- function(dim = 1) {
  keras_model_sequential(input_shape = dim, name = "seq_disc") |> 
    layer_dense(units = 32, activation = "relu") |>
    layer_dense(units = 16, activation = "relu") |>
    layer_dense(units = 1, activation = "sigmoid")
}

summary(generator_mixture())
summary(discriminator_mixture())

# Train
result2 <- train_gan(
  train_dat = train_dat,
  generator_fn = generator_mixture,
  discriminator_fn = discriminator_mixture,
  n_iter = 5,
  epochs_per_iter = 100,
  num_resamples = 1000,
  plot_title = "GAN Training - Mixture",
  true_density_fn = function() {
    curve(0.5 * dnorm(x, 2, 1) + 0.5 * dnorm(x, 8, 2), add = TRUE)
  }
)

# Compare results
cat(sprintf("\nUnimodal training: %.2f seconds\n", result1$time))
cat(sprintf("Mixture training: %.2f seconds\n", result2$time))