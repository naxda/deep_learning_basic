#keras.datasets모듈에서 imdb객체를 import한다.
from keras.datasets import imdb

#훈련 데이터에서 자주 나오는 단어 1만개만 사용, imdb데이터셋을 로드한다.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

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

def vectorize_sequences(sequences, dimension=10000):
    type(sequences)
    print('1111, sequences len, typw : {0}, {1}'.format(len(sequences), type(sequences)))
    results = np.zeros((len(sequences), dimension))
    print('2222, sequences len, dimension, results len, sequence[0] len : {0}, {1}, {2}, {3}'.format(sequences.shape,\
                                                dimension, results.shape, len(sequences[0])))
    for i, sequence in enumerate(sequences):
        print(i)
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print(x_train[0])
print(x_train.shape)
