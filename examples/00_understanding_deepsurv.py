import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, ActivityRegularization
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.regularizers import l2
from lifelines import utils
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import h5py


example_file = '00_understanding_deepsurv'
PATH_DATA = pathlib.Path(r'../deepsurvk/datasets/data')
PATH_MODELS = pathlib.Path('./models/')

# Make sure data directory exists.
if not PATH_DATA.exists():
    raise ValueError(f"The directory {PATH_DATA} does not exist.")

# If models directory does not exist, create it.
if not PATH_MODELS.exists():
    print("Creating models directory in " + str(PATH_MODELS) + "...\t", end="", flush=True)
    PATH_MODELS.mkdir(parents=True)
    print("DONE!")



# %% [markdown]
# ## Get data
# In this case, we will use the Worcester Heart Attack Study (WHAS) dataset.
# For a more detailed description about it, please see the corresponding
# [README](../data/README.md).

# %%
path_data_file = PATH_DATA/'whas.h5'
# X_train: 데이터
# E_train: Deadstatus.event
# Y_train: Survival days

# Read training data.
# with h5py.File(path_data_file, 'r') as f:
#     X_train = f['train']['x'][()]
#     E_train = f['train']['e'][()]
#     Y_train = f['train']['t'][()].reshape(-1, 1)

# Read testing data.
# with h5py.File(path_data_file, 'r') as f:
#     X_test = f['test']['x'][()]
#     E_test = f['test']['e'][()]
#     Y_test = f['test']['t'][()].reshape(-1, 1)


X_train = np.load("../deepsurvk/datasets/X_train_1000.npy")
E_train = np.load("../deepsurvk/datasets/E_train_1000.npy")
Y_train = np.load("../deepsurvk/datasets/Y_train_1000.npy")

X_test = np.load("../deepsurvk/datasets/X_test_200.npy")
E_test = np.load("../deepsurvk/datasets/E_test_200.npy")
Y_test = np.load("../deepsurvk/datasets/Y_test_200.npy")


# Calculate important parameters.
n_patients_train = X_train.shape[0]
n_features = X_train.shape[1]


# %% [markdown]
# ## Pre-process data
# * Standardization <br>
# First, we need to standardize the input (p. 3).
# Notice how we only use training data for the standardization.
# This done to avoid leakage (using information from
# the testing partition for the model training.)

# %%
# X_scaler = StandardScaler().fit(X_train)
# X_train = X_scaler.transform(X_train)
# X_test = X_scaler.transform(X_test)
#
# Y_scaler = StandardScaler().fit(Y_train.reshape(-1, 1))
# Y_train = Y_scaler.transform(Y_train)
# Y_test = Y_scaler.transform(Y_test)

Y_train = Y_train.flatten()
Y_test = Y_test.flatten()

# %% [markdown]
# * Sorting <br>
# This is important, since we are performing a ranking task.

# %%
sort_idx = np.argsort(Y_train)[::-1]
X_train = X_train[sort_idx]
Y_train = Y_train[sort_idx]
E_train = E_train[sort_idx]


# %% [markdown]
# ## Define the loss function
# DeepSurv's loss function is the average negative log partial likelihood with
# regularization (Eq. 4, p. 3):
#
# $$l_{\theta} = -\frac{1}{N_{E=1}} \sum_{i:E_i=1} \left( \hat{h}_\theta(x_i) -\log \sum_{j \in {\rm I\!R}(T_i)} \exp^{\hat{h}_\theta(x_j)} \right) + \lambda \cdot \Vert \theta \Vert_2^2 $$
#
# We can see that our loss function depends on three parameters:
# `y_true`, `y_pred`, *and* `E`. Unfortunately, custom loss functions in Keras
# [need to have their signature (i.e., prototype) as](https://keras.io/api/losses/#creating-custom-losses)
# `loss_fn(y_true, y_pred)`. To overcome this, we will use a [small trick](https://github.com/keras-team/keras/issues/2121)
# that is actually well known in the community. This way, we can define the
# negative log likelihood function as

# %%
def negative_log_likelihood(E):
    def loss(y_true, y_pred):

        hazard_ratio = tf.math.exp(y_pred)
        log_risk = tf.math.log(tf.math.cumsum(hazard_ratio))
        uncensored_likelihood = tf.transpose(y_pred) - log_risk
        censored_likelihood = uncensored_likelihood * E
        neg_likelihood_ = -tf.math.reduce_sum(censored_likelihood)

        # TODO
        # For some reason, adding num_observed_events does not work.
        # Therefore, for now we will use it as a simple factor of 1.
        # Is it really needed? Isn't it just a scaling factor?
        # num_observed_events = tf.math.cumsum(E)
        # num_observed_events = tf.cast(num_observed_events, dtype=tf.float32)
        num_observed_events = tf.constant(1, dtype=tf.float32)
        neg_likelihood = neg_likelihood_ / num_observed_events
        return neg_likelihood
    return loss


activation = 'relu'
n_nodes = 48
learning_rate = 0.067
l2_reg = 16.094
dropout = 0.147
lr_decay = 6.494e-4
momentum = 0.863


# Create model
model = Sequential()
model.add(Dense(units=n_features, activation=activation, kernel_initializer='glorot_uniform', input_shape=(n_features,)))
model.add(Dropout(dropout))
model.add(Dense(units=n_nodes, activation=activation, kernel_initializer='glorot_uniform'))
model.add(Dropout(dropout))
model.add(Dense(units=n_nodes, activation=activation, kernel_initializer='glorot_uniform'))
model.add(Dropout(dropout))
model.add(Dense(units=1, activation='linear', kernel_initializer='glorot_uniform', kernel_regularizer=l2(l2_reg)))
model.add(ActivityRegularization(l2=l2_reg))

# Define the optimizer
# Nadam is Adam + Nesterov momentum
# optimizer = Nadam(learning_rate=learning_rate, decay=lr_decay, clipnorm=1)
optimizer = Nadam(learning_rate=learning_rate, weight_decay=lr_decay)

# Compile the model and show a summary of it
model.compile(loss=negative_log_likelihood(E_train), optimizer=optimizer)
model.summary()


# %% [markdown]
# Sometimes, the computation of the loss yields a `NaN`, which makes the whole
# output be `NaN` as well. I haven't identified a pattern, actually I think
# it is quite random. This could be due to a variety of reasons, including
# model parametrization (however, I don't really want to use different
# parameters than those reported), maybe even unfortunate parameter
# initialization. Therefore, we will use a technique called "Early Stopping".
#
# In this case, we will train the model until the number of epochs is reached
# *or* until the loss is an `NaN`. After that, training is stopped. Then,
# we will selected and use the model that yielded the smallest lost.
#
# We can achieve this very easily using [callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback)

# %%
callbacks = [tf.keras.callbacks.TerminateOnNaN(),
             tf.keras.callbacks.ModelCheckpoint(str(PATH_MODELS/f'{example_file}.h5'), monitor='loss', save_best_only=True, mode='min')]

# %% [markdown]
# ## Model fitting
# Now we can fit the DeepSurv model. Notice how we use the whole set of
# patients in a batch. Furthermore, be sure that `shuffle` is set to `False`,
# since order is important in predicting ranked survival.

epochs = 200
history = model.fit(X_train, Y_train,
                    batch_size=n_patients_train,
                    epochs=epochs,
                    callbacks=callbacks,
                    shuffle=False)

# fig, ax = plt.subplots(1, 1, figsize=[5, 5])
# plt.plot(history.history['loss'], label='train')
# ax.set_xlabel("No. epochs")
# ax.set_ylabel("Loss [u.a.]")


model = load_model(PATH_MODELS/f'{example_file}.h5', compile=False)

Y_pred_train = np.exp(-model.predict(X_train))
c_index_train = utils.concordance_index(Y_train, Y_pred_train, E_train)
print(f"c-index of training dataset = {c_index_train}")

Y_pred_test = np.exp(-model.predict(X_test))
c_index_test = utils.concordance_index(Y_test, Y_pred_test, E_test)
print(f"c-index of testing dataset = {c_index_test}")
