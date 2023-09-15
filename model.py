import numpy as np
import re

class Model:
    def __init__(self, alpha=1):
        self.vocab = set() # словарь, содержащий все уникальные слова из набора train
        self.spam = {} # словарь, содержащий частоту слов в спам-сообщениях из набора данных train.
        self.ham = {} # словарь, содержащий частоту слов в не спам-сообщениях из набора данных train.
        self.alpha = alpha # сглаживание
        self.label2num = None # словарь, используемый для преобразования меток в числа
        self.num2label = None # словарь, используемый для преобразования числа в метки
        self.Nvoc = None # общее количество уникальных слов в наборе данных train
        self.Nspam = None # общее количество уникальных слов в спам-сообщениях в наборе данных train
        self.Nham = None # общее количество уникальных слов в не спам-сообщениях в наборе данных train
        self._train_X, self._train_y = None, None
        self._val_X, self._val_y = None, None
        self._test_X, self._test_y = None, None

    def fit(self, dataset):
        '''
        dataset - объект класса Dataset
        Функция использует входной аргумент "dataset", 
        чтобы заполнить все атрибуты данного класса.
        '''
        # Начало вашего кода
        self.label2num={'ham': 0, 'spam':1}
        self.num2label={0 : 'ham', 1 : 'spam'}
        
        self._train_X = dataset.train[0]
        self._train_y = dataset.train[1]
        
        self._val_X = dataset.val[0]
        self._val_y = dataset.val[1]
        
        self._test_X = dataset.test[0]
        self._test_y = dataset.test[1]
        
        for i in self._train_X:
            i=i.split(' ')
            for j in i:
                self.vocab.add(j)   #here fill vocab
                
        index_of_spam = np.where(self._train_y == 1)
        train_spam = self._train_X[index_of_spam]
        
        index_of_ham = np.where(self._train_y == 0)
        train_ham = self._train_X[index_of_ham]
        
        for i in range(len(train_spam)):
            element=train_spam[i].split(' ')
            for j in element:
                if not j in self.spam:
                    self.spam[j]=0
                self.spam[j]+=1
        self.spam.pop('')    
                    
            
        for i in range(len(train_ham)):
            element=train_ham[i].split(' ')
            for j in element:
                if not j in self.ham:
                    self.ham[j]=0
                self.ham[j]+=1
        self.ham.pop('')    
        
        
        self.Nvoc = len(self.vocab)
        self.Nspam = len(self.spam.keys())
        self.Nham = len(self.ham.keys())
        
        # Конец вашего кода
        pass
    
    def inference(self, message):
        '''
        Функция принимает одно сообщение и, используя наивный байесовский алгоритм, определяет его как спам / не спам.
        '''
        # Начало вашего кода
        def calc_probs(sam_vocab, message, alpha=1):
            #Начало вашего кода    
            message=re.sub(' +', ' ', re.sub(r'[^\w]', ' ', message.lower())).split(' ')
    
            if '' in message:
                message.remove('')  
    
            res=1
 
            for i in range(len(message)):
                if message[i] in sam_vocab.keys():
                    t=sam_vocab[message[i]]
                else:
                    t=0
                res*=(t + alpha) / (sum(sam_vocab.values()) + alpha * self.Nvoc)           
   
            return res
        
        prob_spam = sum(self._train_y) / len(self._train_y)
        prob_ham = 1 - prob_spam
        
        pspam = prob_spam * calc_probs(self.spam, message)
        pham = prob_ham * calc_probs(self.ham, message)
        
        # Конец вашего кода
        if pspam > pham:
            return "spam"
        return "ham"
    
    def validation(self):
        '''
        Функция предсказывает метки сообщений из набора данных validation,
        и возвращает точность предсказания меток сообщений.
        Вы должны использовать метод класса inference().'''
        
        # Начало вашего кода
        val_acc_list = []
        for i in self._val_X:
            res = Model.inference(self, message = i)
            val_acc_list.append(self.label2num[res])
            
        val_acc = sum(np.array(val_acc_list) == self._val_y) / len(self._val_y)
        # Конец вашего кода
        return val_acc 

    def test(self):
        '''
        Функция предсказывает метки сообщений из набора данных test,
        и возвращает точность предсказания меток сообщений.
        Вы должны использовать метод класса inference().
        '''
        # Начало вашего кода
        test_acc_list = []
        for i in self._test_X:
            res = Model.inference(self, message = i)
            test_acc_list.append(self.label2num[res])
            
        test_acc = sum(np.array(test_acc_list) == self._test_y) / len(self._test_y)
        # Конец вашего кода
        return test_acc


