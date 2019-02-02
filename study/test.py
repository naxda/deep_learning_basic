#keras.datasets모듈에서 imdb객체를 import한다.
from keras.datasets import imdb

#훈련 데이터에서 자주 나오는 단어 1만개만 사용, imdb데이터셋을 로드한다.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)

#당연하지만.. 최대 인덱스는 9999까지다..
max_index = max([max(sequence) for sequence in train_data])
print(max_index)

#
word_index = imdb.get_word_index()#word_index는 단어와 정수 인덱스를 매핑한 딕셔너리
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]#정수인덱스, 단어 순서로 다시 저장한다.
)
decoded_index = ' '.join(
    [reverse_word_index.get(i-3, '?') for i in train_data[0]]
)

print(train_data[0])

import numpy as np

def vectorize_sequences(sequences, dimension=1000):
    type(sequences)
    print('1111, sequences len, type : {0}, {1}'.format(len(sequences), type(sequences)))
    results = np.zeros((len(sequences), dimension))
    print('2222, sequences len, dimension, results len, sequence[0] len : {0}, {1}, {2}, {3}'.format(sequences.shape,\
                                                dimension, results.shape, len(sequences[0])))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print(x_train[0])
# print(x_train.shape)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# model 정의
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(1000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# 옵티마이저 설정
from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy'])

# 훈련 검증
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

# 모델 훈련
model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 훈련과 검증 손실 그리기
import matplotlib.pyplot as plt

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validataion loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# 훈련과 검증 정확도 그리기
plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# 모델을 처음부터 다시 훈련하기 -> 왜 처음부터 다시 훈련?
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(1000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# 훈련된 모델로 새로운 데이터 예측하기
model.predict(x_test)

print('end !!!!')