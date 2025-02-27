original_dim = 784
vmod = vae.VariationalAutoEncoder(original_dim, 64, 32)

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = keras.losses.MeanSquaredError()

loss_metric = keras.metrics.Mean()

# (x_train, _), _ = keras.datasets.mnist.load_data()
# x_train = x_train.reshape(60000, 784).astype("float32") / 255

# train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
# train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

x_train = X1.iloc[:512, :original_dim].to_numpy().astype("float32")
train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

epochs = 2

# Iterate over epochs.
for epoch in range(epochs):
    print("Start of epoch %d" % (epoch,))
    # Iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = vmod(x_batch_train)
            # Compute reconstruction loss
            loss = mse_loss_fn(x_batch_train, reconstructed)
            loss += sum(vmod.losses)  # Add KLD regularization loss
        grads = tape.gradient(loss, vmod.trainable_weights)
        optimizer.apply_gradients(zip(grads, vmod.trainable_weights))
        loss_metric(loss)
        if step % 100 == 0:
            print("step %d: mean loss = %.4f" % (step, loss_metric.result()))


