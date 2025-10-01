library(keras3)
library(tensorflow)

# https://thomvolker.github.io/blog/1407_gans_in_r/

set.seed(123)
set_random_seed(123)

library(compiler)
enableJIT(3) # Level 3 is the highest

# 1 - unimodal ----

N <- 1000
train_dat <- matrix(rnorm(n=N, mean=2, sd=4))

plot(density(train_dat), col = "darkorange2")
curve(dnorm(x, 2, 4), add = TRUE)

generator <- function(latent_dim = 1) {
  model <- keras_model_sequential(input_shape = latent_dim, name = "seq_gen") |> 
    layer_dense(units = 1, activation = "linear")
}

summary(generator())

discriminator <- function(dim = 1) {
  model <- keras_model_sequential(input_shape = dim, name = "seq_disc") |> 
    layer_dense(units = 8, activation = "relu") |>
    layer_dense(units = 1, activation = "sigmoid")
}

summary(discriminator())

gan <- keras3::new_model_class(
  classname = "GAN",
  # initialize model with generator, discriminator, dimension
  # of the random latent vectors (i.e., the input that is 
  # transformed by the generator to yield useful synthetic 
  # data).
  initialize = function(discriminator, generator, latent_dim) {
    super$initialize()
    self$discriminator <- discriminator
    self$generator <- generator
    self$latent_dim <- latent_dim
    self$d_loss_metric <- metric_mean(name = "d_loss")
    self$g_loss_metric <- metric_mean(name = "g_loss")
  },
  # create compile function that sets the optimizers and loss
  compile = function(d_optimizer, g_optimizer, loss_fn) {
    super$compile()
    self$d_optimizer <- d_optimizer
    self$g_optimizer <- g_optimizer
    self$loss_fn <- loss_fn
  },
  # plot generator and discriminator loss during training
  metrics = mark_active(function() {
    list(self$d_loss_metric, self$g_loss_metric)
  }),
  # define the training step, set batch size, create random normal variates
  # as input for the generator, stack real and generated data, create labels
  # for the discriminator, add some noise to the labels to prevent overfitting,
  # compute discriminator loss, compute gradients, apply gradients to the
  # discriminator.
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
    
    # Then sample new random points in latent space, and create labels as if all
    # these new samples were real so that only the generator is trained, and not
    # the discriminator. Then the generator loss is computed, and the generator 
    # weights are updated. 
    
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

latent_dim <- as.integer(1)
nsyn <- nrow(train_dat)

mod <- gan(
  discriminator = discriminator(dim = ncol(train_dat)), 
  generator = generator(latent_dim = latent_dim), 
  latent_dim = latent_dim
)

mod |>
  compile(
    d_optimizer = optimizer_adam(beta_1 = 0.5),
    g_optimizer = optimizer_adam(beta_1 = 0.5),
    loss_fn = loss_binary_crossentropy()
  )

plot(
  density(train_dat), 
  ylim = c(0, 0.2), 
  xlim = c(-12, 16), 
  col = "darkorange2",
  xlab = "x",
  main = "GAN Training"
)

curve(dnorm(x, 2, 4), add = TRUE)
curve(dnorm(x, 0, 1), add = TRUE, col = "black", lty=2)

n_iter <- 5
start <- proc.time()[3]
pb <- utils::txtProgressBar(max=n_iter, style = 3L)
for (i in seq_len(n_iter)) {
  mod |>
    fit(train_dat, epochs = 30, batch_size = 32, verbose = 0)
  newdat <- mod$generator(tf$random$normal(shape = c(nsyn, latent_dim)))
  lines(density(as.matrix(newdat)), 
        col = RColorBrewer::brewer.pal(7, "Greens")[i])
  utils::setTxtProgressBar(pb, i)
}
paste0("Elapsed ", proc.time()[3] - start)
close(pb)

# Number of new resamples you want to generate
num_resamples <- 500  # Adjust this number as needed

# Generate random latent vectors (input noise)
#latent_vectors <- tf$random$normal(shape = c(num_resamples, latent_dim))
latent_vectors <- matrix(rnorm(n=num_resamples*latent_dim), 
                         nrow = num_resamples, 
                         ncol = latent_dim)

# Get the synthetic data by passing latent vectors through the generator
resamples <- mod$generator(latent_vectors)

# Convert the result to an R matrix or array
resamples_matrix <- as.matrix(resamples)

# Plot the density of the generated resamples along with the real data
plot(density(train_dat), col = "darkorange2", 
     xlim = c(-12, 16), ylim = c(0, 0.2), 
     xlab = "x", main = "Generated Resamples vs Real Data")

# Plot the density of the generated resamples
lines(density(resamples_matrix), col = "darkgreen", lty = 2)

# Optionally, you can also visualize the comparison in histograms
hist(train_dat, col = rgb(1, 0.5, 0, 0.5), xlim = c(-12, 16), main = "Histogram of Real vs Generated Data", 
     xlab = "x", probability = TRUE)
hist(resamples_matrix, col = rgb(0, 0.5, 0, 0.5), add = TRUE, probability = TRUE)
legend("topright", legend = c("Real Data", "Generated Data"), 
       fill = c(rgb(1, 0.5, 0, 0.5), rgb(0, 0.5, 0, 0.5)))


# 2 - mixture ----

set.seed(123)

mixture <- rbinom(N, 1, 0.5)
train_dat <- matrix(mixture * rnorm(N, 2, 1) + (1 - mixture) * rnorm(N, 8, 2))

plot(density(train_dat), ylim = c(0, 0.2), col = "darkorange2")
curve(0.5 * dnorm(x, 2, 1) + 0.5 * dnorm(x, 8, 2), add = TRUE)

generator <- function(latent_dim = 1) {
  model <- keras_model_sequential(input_shape = latent_dim, name = "seq_gen") |> 
    layer_dense(units = 16, activation = "relu") |>
    layer_dense(units = 1, activation = "linear")
}

summary(generator())

discriminator <- function(dim = 1) {
  model <- keras_model_sequential(input_shape = dim, name = "seq_disc") |> 
    layer_dense(units = 32, activation = "relu") |>
    layer_dense(units = 16, activation = "relu") |>
    layer_dense(units = 1, activation = "sigmoid")
}

summary(discriminator())

latent_dim <- as.integer(1)
nsyn <- nrow(train_dat)

mod <- gan(
  discriminator = discriminator(dim = ncol(train_dat)), 
  generator = generator(latent_dim = latent_dim), 
  latent_dim = latent_dim
)

mod |>
  compile(
    d_optimizer = optimizer_adam(beta_1 = 0.5),
    g_optimizer = optimizer_adam(beta_1 = 0.5),
    loss_fn = loss_binary_crossentropy()
  )

plot(
  density(train_dat), 
  ylim = c(0, 0.2), 
  col = "darkorange2",
  xlab = "x",
  main = "GAN Training"
)
curve(0.5 * dnorm(x, 2, 1) + 0.5 * dnorm(x, 8, 2), add = TRUE)

n_iter <- 5
pb <- utils::txtProgressBar(max=n_iter, style = 3L)
start <- proc.time()[3]
for (i in seq_len(n_iter)) {
  mod |>
    fit(train_dat, epochs = 100, batch_size = 32, verbose = 0)
  newdat <- mod$generator(tf$random$normal(shape = c(nsyn, latent_dim)))
  lines(density(as.matrix(newdat)), 
        col = RColorBrewer::brewer.pal(6, "Greens")[i])
  utils::setTxtProgressBar(pb, i)
}
paste0("Elapsed ", proc.time()[3] - start)
close(pb)




# Number of new resamples you want to generate
num_resamples <- 1000  # Adjust this as needed

# Generate random latent vectors (input noise)
#latent_vectors <- tf$random$normal(shape = c(num_resamples, latent_dim))
latent_vectors <- matrix(rnorm(n=num_resamples*latent_dim), 
                         nrow = num_resamples, 
                         ncol = latent_dim)

# Get the synthetic data by passing latent vectors through the generator
resamples <- mod$generator(latent_vectors)

# Convert the result to an R matrix or array
resamples_matrix <- as.matrix(resamples)

# Plot the density of the generated resamples along with the real data
plot(density(train_dat), col = "darkorange2", 
     xlim = c(-12, 16), ylim = c(0, 0.2), 
     xlab = "x", main = "Generated Resamples vs Real Data")

# Plot the density of the generated resamples
lines(density(resamples_matrix), col = "darkgreen", lty = 2)

# Optionally, you can also visualize the comparison in histograms
hist(train_dat, col = rgb(1, 0.5, 0, 0.5), xlim = c(-12, 16), main = "Histogram of Real vs Generated Data", 
     xlab = "x", probability = TRUE)
hist(resamples_matrix, col = rgb(0, 0.5, 0, 0.5), add = TRUE, probability = TRUE)
legend("topright", legend = c("Real Data", "Generated Data"), 
       fill = c(rgb(1, 0.5, 0, 0.5), rgb(0, 0.5, 0, 0.5)))
