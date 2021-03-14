import tensorflow as tf
#dataset

"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
#this code is only required if we run more than 1 GPU for training.
"""


class report_accuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= 0.9:
            print('\nReaching {} accuracy. Terminating training loop.'.format(logs.get('accuracy')))
            self.model.stop_training = True
report_accuracy_object = report_accuracy()

f.keras.datasets.fashion_mnist
dataset = t
(images, labels) , (testimages, testlabels) = dataset.load_data()

images, testimages = images/255.0, testimages/255.0

images = images.reshape(images.shape[0], 28, 28, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=(28,28,1), filters = 10, kernel_size=(2,2), strides= (1,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(28*28*10, activation= tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='Adam', loss ='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(images, labels, epochs = 1, batch_size=2048)

'''
there could be a warning of Callback method 'on_train_batch_end' is slow compared to the batch time.
This is caused when other operations which runs at the end of each batch consumes longer time than batch themselves.
One of the causes might be, we have really smaller batches. So, any operations is slow compared to your original batches.
we can also have this warning because of tensorboard callback.Tensorboard's callback took longer for saving and writing log files 
than an iteration of the batch size (let's say 32 examples). We can remove Tensorboard from callbacks' list and the Warning should disappear'
'''


