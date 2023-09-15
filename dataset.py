import numpy as np
import re

class Dataset:
    def __init__(self, X, y):
        self._x = X # сообщения 
        self._y = y # метки ["spam", "ham"]
        self.train = None # кортеж из (X_train, y_train)
        self.val = None # кортеж из (X_val, y_val)
        self.test = None # кортеж из (X_test, y_test)
        self.label2num = {} # словарь, используемый для преобразования меток в числа
        self.num2label = {} # словарь, используемый для преобразования числа в метки
        self._transform()
        
    def __len__(self):
        return len(self._x)
    
    def _transform(self):
        '''
        Функция очистки сообщения и преобразования меток в числа.
        '''
        # Начало вашего кода
        self.label2num={'ham': 0, 'spam':1} 
        
        for i in range(len(self._x)):
            self._x[i]=re.sub(' +', ' ', re.sub(r'[^\w]', ' ', self._x[i].lower()))
            self._y[i]=self.label2num[self._y[i]]
        # Конец вашего кода
        
        pass

    def split_dataset(self, val=0.1, test=0.1):
        '''
        Функция, которая разбивает набор данных на наборы train-validation-test.
        '''
        self.num2label={0 : 'ham', 1 : 'spam'}
        # Начало вашего кода
        np.random.seed(1)
        indices = np.arange(0, len(self._x), 1)
        np.random.shuffle(indices)
        
        val_indices = indices[:round(len(indices)*val)]
        self.val=(self._x[val_indices], self._y[val_indices])
        
        test_indices = indices[round(len(indices)*val) : round(len(indices)*(val+test))]
        self.test=(self._x[test_indices], self._y[test_indices])
        
        train_indices = indices[round(len(indices)*(val+test)) :]
        self.train=(self._x[train_indices], self._y[train_indices])
        
        return {'train': self.train, 'val': self.val, 'test': self.test}
        # Конец вашего кода
        pass

