import matplotlib.pyplot as plt
import numpy as np
import glob
import PIL
import os
import time
#from silence_tf import tensorflow_shutup
#tensorflow_shutup()
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model



"""
Going to be following tensorflow tutorial for this one
https://www.tensorflow.org/tutorials/generative/dcgan
"""

BUFFER_SIZE = 60000
BATCH_SIZE = 256
# training setup
EPOCHS = 100



(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
#train_images = train_images[:, :, :, None]
train_images = train_images.reshape(
                    train_images.shape[0], 28, 28, 1).astype('float32')         # this one is from tutorial, feels more unclear
train_images = (train_images - 127.5) / 127.5                                   # weird way to normalize
#plt.imshow(train_images[0], cmap='Greys')
#plt.show()
train_dataset = tf.data.Dataset.from_tensor_slices(
                    train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# helper object, this implementation could be cleaned up a bit
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# 16 'images' so we can compare the progress over epochs
noise_dimension = 100
n_examples_to_generate = 16
seed_images = tf.random.normal([n_examples_to_generate, noise_dimension])

# setting up optimizers for generator and discriminator
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)



def generator_model():
    """
    Discription grabbed from tutorial:

    The generator uses tf.keras.layers.Conv2DTranspose (upsampling) layers
    to produce an image from a seed (random noise). Start with a Dense layer
    that takes this seed as input, then upsample several times until you reach
    the desired image size of 28x28x1. Notice the tf.keras.layers.LeakyReLU
    activation for each layer, except the output layer which uses tanh.
    """

    # We define our model. From Keras: Sequential groups a linear stack of
    #                       of layers into a tf.keras.Model. Sequential provides
    #                       training and inference features on this model.
    model = tf.keras.Sequential()

    # Next we add the layers we want to our model. Here is where most of the
    # "magic" comes into play. Knowing which layers to combine seems like a
    # difficult problem with no real 'standard'.

    # Dense means that every neuron is connected and the input shape is the
    # shape of our random noise to seed our generator
    model.add(layers.Dense(units=7*7*256, use_bias=False, input_shape=(100,)))  # I think this number 12544 is sort of arbitrary
    # We normalize the outputs from the Dense layer
    model.add(layers.BatchNormalization())
    # And add a Leaky ReLU activation function to our whole 'layer'
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))                                      # Number here matches with above. Not sure if it has something to do with wanting to end up at an mnist sized image or what.
    assert model.output_shape == (None, 7, 7, 256)
    # Even though we just added four keras layers we think of this chunk of code
    # above as one layer

    # Next we add our upscaling layers. Probably best to read about the params
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose
    model.add(layers.Conv2DTranspose(filters=128,
                                     kernel_size=(5, 5),
                                     strides=(1, 1),
                                     padding='same',
                                     use_bias=False))

    # Then we add the same normalization and activation
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 7, 7, 128)

    model.add(layers.Conv2DTranspose(filters=64,
                                     kernel_size=(5, 5),
                                     strides=(2, 2),
                                     padding='same',
                                     use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 14, 14, 64)


    # The strides need to match up corresponding to the jump between input dims
    # I tried skipping one upscaling from (7, 7, 128) -> (14, 14, 64)
    # however this produced uninteresting results
    model.add(layers.Conv2DTranspose(filters=1,
                                     kernel_size=(5, 5),
                                     strides=(2, 2),
                                     padding='same',
                                     use_bias=False,
                                     activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model



def discriminator_model():
    """
    The discriminator is a convolutional neural network based image classifier
    """
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=64,
                            kernel_size=(5, 5),
                            strides=(2, 2),
                            padding='same',
                            input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    # adding a dropout layer as you do in conv-nets
    model.add(layers.Dropout(0.3))


    model.add(layers.Conv2D(filters=128,
                            kernel_size=(5, 5),
                            strides=(2, 2),
                            padding='same'))
    model.add(layers.LeakyReLU())
    # adding a dropout layer as you do in conv-nets
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model



generator = generator_model()
discriminator = discriminator_model()


# Setting up checkpoints to save model during training
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                            discriminator_optimizer=discriminator_optimizer,
                            generator=generator,
                            discriminator=discriminator)



def generator_loss(fake_output):
    loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return loss



def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss



# Notice the use of tf.function here, it causes the whole function to be
# 'compiled'. This replaces how we would use tf.session in tf 1.x

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dimension])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss,
                                            generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                            discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                            discriminator.trainable_variables))

    return gen_loss, disc_loss



def generate_and_save_images(model, epoch, test_input):
    # we're making inferences here
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(f'./images_from_seed_images/image_at_epoch_{str(epoch).zfill(3)}.png')
    plt.close()
    #plt.show()



def train(dataset, epochs):
    generator_loss_list = []
    discriminator_loss_list = []

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            generator_loss_list.append(gen_loss.numpy())
            discriminator_loss_list.append(disc_loss.numpy())

        generate_and_save_images(generator, epoch + 1, seed_images)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'Time for epoch {epoch} is {time.time() - start}')

    generate_and_save_images(generator, epochs, seed_images)


    loss_file = './data/lossfile.txt'
    with open(loss_file, 'w') as outfile:
        outfile.write(str(generator_loss_list))
        outfile.write('\n')
        outfile.write('\n')
        outfile.write(str(discriminator_loss_list))
        outfile.write('\n')
        outfile.write('\n')



train(train_dataset, EPOCHS)





"""
# Testing to see how it works
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
decision = discriminator(generated_image)
print(decision)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()

generator_plot_file = './model_plots/generator.png'
discriminator_plot_file = './model_plots/discriminator.png'
plot_model(generator, to_file=generator_plot_file, show_shapes=True)
plot_model(discriminator, to_file=discriminator_plot_file, show_shapes=True)
#"""
