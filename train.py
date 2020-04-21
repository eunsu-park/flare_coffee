gpu_id = 0
save_frequenty = 10
max_epoch = 100
ch_input = 1
ch_output = 2

isize = 256
growth_rate=16
nb_module=2
nb_block=5

bsize = 32

root_data    = 'path_to_data'
root_save    = 'path_to_save'

import tensorflow as tf
import tensorflow.keras as keras

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
tf.config.experimental.set_memory_growth(gpus[gpu_id], True)

from networks import coffee
model = coffee(isize, ch_input, ch_output, growth_rate, nb_module, nb_block)
model.summary()

optimizer = keras.optimizers.Adam(lr=2.e-4)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_batch_generator = ImageDataGenerator(rescale=1./255., validation_split=0.1)
test_batch_generator = ImageDataGenerator(rescale=1./255.)

train_batch = train_batch_generator.flow_from_directory('%s/train'%(root_data),
                                                        target_size=(isize, isize),
                                                        batch_size = bsize,
                                                        class_mode='categorical',
                                                        color_mode=color_mode,
                                                        subset='train')

validation_batch = train_batch_generator.flow_from_directory('%s/train'%(root_data),
                                                             target_size=(isize, isize),
                                                             batch_size = bsize,
                                                             class_mode='categorical',
                                                             color_mode=color_mode,
                                                             subset='validation')

test_batch = test_batch_generator.flow_from_directory('%s/test'%(root)data,
                                                      target_size=(isize, isize),
                                                      batch_size=bsize,
                                                      class_mode='categorical',
                                                      color_mode=color_mode,
                                                      shuffle=False)
                                                      

nb_train = train_batch.n
nb_validation = validation_batch.n
nb_test = test_batch.n

nb_batch_train = nb_training//bsize
nb_batch_validation = nb_validation//bsize
nb_batch_test = nb_test//bsize

class custom_history:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs = {}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

custom_callback = custom_history(0
custom_callback.init()

epoch = 0
while epoch <= max_epoch

    model.fit_generator(train_batch, nb_batch_train, validation_data=validation_batch,
                        epochs=save_frequency, callbacks = [custom_history])

    epoch += save_frequency

    model.save('%s/epoch_%d.h5'%(root_save, epoch))

import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(custom_history.train_loss, 'y', label='train loss')
loss_ax.plot(custom_history.val_loss, 'r', label='val loss')
acc_ax.plot(custom_history.train_acc, 'b', label='train acc')
acc_ax.plot(custom_history.val_acc, 'g', label='val_acc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')
loss_ax.legend(loc='upper left')
acc_ac.legend(loc='lower left')
fig.savefit('%s/plot.png'%(root_save))


