## Sequentialクラスを使って定義した２層のモデル
#from keras import models
#from keras import layers
#model = models.Sequential()
#model.add(layers.Dense(32, activation='relu', input_shape=(784,)))
#model.add(layers.Dense(10, activation='softmax'))

#from keras import optimisers
#model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='use', metrics=['accuracy'])

#model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)

# リスト3.1
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
num_words=10000)

print(train_data[0])
print(train_labels[0])
print(max([max(sequence) for sequence in train_data]))

# word_indexは単語を整数のインデックスにマッピングする辞書
word_index = imdb.get_word_index()
# 整数のインデックスを単語にマッピング
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# レビューをデコード：インデックスのオフセットとして3が指定されているのは、
# 0、1、2がそれぞの「パディング」、「シーケンスの開始」、「不明」の
# インデックスとして予約されているためであることに注意
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_review)

# リスト3.2
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # 形状が(len(sequences), dimension)の行列を作成し、0で埋める
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # results[i]のインデックスを1に設定
    return results

# 訓練データのベクトル化
x_train = vectorize_sequences(train_data)
# テストデータのベクトル化
x_test = vectorize_sequences(test_data)

print(x_train[0])

# ラベルもベクトル化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# リスト3.3　モデルの定義
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# リスト3.4　損失関数とオプティマイザ

#model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])

#from keras import losses
#from keras import metrics

#model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#              loss=losses.binary_crossentropy,
#              metrics=[metrics.binary_accuracy])

# リスト3.7　検証データセットの設定
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# リスト3.8　モデルの訓練
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['acc'])

history = model.fit(partial_x_train,
partial_y_train,
epochs=20,
batch_size=512,
validation_data=(x_val, y_val))

history_dict = history.history
history_dict.keys()

# リスト3.9　訓練データと検証データでの損失値をプロット
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
# “bo”は“（青のドット）を意味する”
plt.plot(epochs, loss, 'bo', label='Training loss')
# “b”は“（青の実線）を意味する”
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# リスト3.10　訓練データと検証データの正解率をプロット
# 図を消す
plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# リスト3.11　モデルの訓練をやり直す
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print(results)

print(model.predict(x_test))
