'''
Author: Harshil Prajapati
'''

from base import *

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization
from keras.losses import binary_crossentropy, mse
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    categorical_inputs = []
    for cat in cat_cols:
        categorical_inputs.append(Input(shape=[1], name=cat))

    categorical_embeddings = []
    for i, cat in enumerate(cat_cols):
        categorical_embeddings.append(
            Embedding(embed_sizes[i], 10)(categorical_inputs[i]))

    # Categorical layer (asset code)
    categorical_logits = Flatten()(categorical_embeddings[0])
    categorical_logits = Dense(64, activation='relu')(categorical_logits)

    # Numeric layer describing assets (returns, std, etc)
    # 18 numeric inputs.  11 from data set, 10 hand engineered.
    numerical_inputs = Input(shape=(23, ), name='num')
    numerical_logits = numerical_inputs
    numerical_logits = BatchNormalization()(numerical_logits)

    numerical_logits = Dense(248, activation='relu')(numerical_logits)
    numerical_logits = Dense(124, activation='relu')(numerical_logits)
    numerical_logits = Dense(64, activation='relu')(numerical_logits)

    # Combine our numeric and catergoic layers:
    logits = Concatenate()([numerical_logits, categorical_logits])
    logits = Dense(64, activation='relu')(logits)
    logits = Dense(32, activation='relu')(logits)
    out = Dense(1, activation='sigmoid')(logits)  # Single output

    model = Model(inputs=categorical_inputs + [numerical_inputs], outputs=out)
    model.compile(optimizer='adam', loss=binary_crossentropy)

    check_point = ModelCheckpoint(
        'model.hdf5', verbose=True, save_best_only=True)
    early_stop = EarlyStopping(patience=12, verbose=True)
    model.fit(X_train, y_train.astype(int),
              validation_data=(X_valid, y_valid.astype(int)),
              epochs=100,
              verbose=True,
              callbacks=[early_stop, check_point])

    model.load_weights('model.hdf5')
    confidence_valid = model.predict(X_valid)[:, 0]*2 - 1
    print(accuracy_score(confidence_valid > 0, y_valid))
    plt.hist(confidence_valid, bins='auto')
    plt.title("predicted confidence")
    plt.show()  # 9.) Some Sanity Checks. Real vs Pred

    # get rid of outliers. Where do they come from??
    r_valid = r_valid.clip(-1, 1)
    x_t_i = confidence_valid * r_valid * u_valid
    data = {'day': d_valid, 'x_t_i': x_t_i}
    df = pd.DataFrame(data)
    x_t = df.groupby('day').sum().values.flatten()
    mean = np.mean(x_t)
    std = np.std(x_t)
    score_valid = mean / std
    print(score_valid)
