import codecs
import os
import collections
from six.moves import cPickle
import numpy as np


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        # os.path 모듈 : 파일 경로를 생성 및 수정하고, 파일 정보를 쉽게 다룰 수 있게 해주는 모듈.
        # os.path.join() : 해당 OS 형식에 맞도록 입력 받은 경로를 연결(입력 중간에 절대경로가 나오면 이전에 취합딘 경로는 제거하고 다시 연결)

        # os.path.join(data_dir, "input.txt") => 'data/Balad_Song/input.txt'
        input_file = os.path.join(data_dir, "input.txt")
        # os.path.join(data_dir, "vocab.pkl") => 'data/Balad_Song/vocab.pkl'
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        # os.path.join(data_dir, "data.npy") => 'data/Balad_Song/data.npy'
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            # preprocess : util.py에서 정의한 함수, line 37 참조
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            # load_preprocessed : util.py에서 정의한 함수, line 52 참조
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    # preprocess data for the first time.
    def preprocess(self, input_file, vocab_file, tensor_file):
        # with - as 구문 : 파일을 열고 닫을 때 close() 메소드 없이 파일을 닫을 수 있음
        # with 블록이 종료될 때 자동으로 파일을 close 시켜주기 때문에 코드를 간단히 줄일 수 있다.
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()

        # collections.Counter() : 컨테이너에 동일한 값의 자료가 몇개인지를 파악하는데 사용하는 객체
        #                       : 리턴 값은 딕셔너리 형태({요소1 : 요소1의 개수, 요소2 : 요소2의 개수, ...})로 출력
        counter = collections.Counter(data)
        # sorted() : iterable한 자료형에 대해 동작함.
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    # load the preprocessed the data if the data has been processed before.
    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    # seperate the whole data into different batches.
    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        # reshape the original data into the length self.num_batches * self.batch_size * self.seq_length for convenience
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)

        # ydata is the xdata with one position shift.
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
