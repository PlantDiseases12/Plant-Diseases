import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam
from Evaluation import evaluation

# -------------------------
# Generator
# -------------------------
def build_generator(latent_dim, input_shape,hidden_neurons):
    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=latent_dim))
    model.add(LeakyReLU(0.2))
    model.add(Dense(np.prod(input_shape), activation='tanh'))
    model.add(Reshape(input_shape))
    return model

# -------------------------
# Discriminator (Classifier)
# -------------------------
def build_discriminator(input_shape, num_classes,hidden_neurons, ):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape,))
    model.add(Dense(hidden_neurons))
    model.add(LeakyReLU(0.2))
    model.add(Dense(num_classes))
    model.add(LeakyReLU(0.2))
    model.add(Dense(23, activation='softmax'))  # +1 for 'fake' class
    return model

# -------------------------
# GAN-based training function
# -------------------------
def Model_GAN(x_train, y_train, x_test, y_test,Hidden_Neuron_Count, epochs=5000, batch_size=32):
    input_shape = x_train.shape[1:]
    latent_dim = 100
    num_classes = y_train.shape[1]


    # Build models
    generator = build_generator(latent_dim, input_shape,Hidden_Neuron_Count)
    discriminator = build_discriminator(input_shape, num_classes,Hidden_Neuron_Count)
    discriminator.compile(loss='BinaryCrossentropy',
                          optimizer=Adam(0.0002, 0.5),
                          metrics=['accuracy'])
    discriminator.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

    # Combined model (Generator + Discriminator)
    z = Input(shape=(latent_dim,))
    fake = generator(z)
    discriminator.trainable = False
    validity = discriminator(fake)
    combined = Model(z, validity)
    combined.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.0002, 0.5))

    # Training Loop
    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_imgs = x_train[idx]
        real_labels = np.argmax(y_train[idx], axis=1)

        # Generate fake images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)

        # Fake labels are 'num_classes' (one above real class indices)
        fake_labels = np.ones((batch_size,)) * num_classes

        # Train discriminator on real and fake
        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake_labels)

        # ---------------------
        #  Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        trick_labels = np.random.randint(0, num_classes, batch_size)
        g_loss = combined.train_on_batch(noise, trick_labels)

        # Print
        if epoch % 100 == 0:
            print(f"{epoch} [D real acc: {d_loss_real[1]*100:.2f}%] [G loss: {g_loss:.4f}]")

    # Evaluation
    y_pred = discriminator.predict(x_test)
    y_pred_classes = np.argmax(y_pred[:, :num_classes], axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Use your custom evaluation
    Eval = evaluation(y_true_classes, y_pred_classes)

    return Eval, y_pred_classes