#!/usr/bin/env python
# coding: utf-8

# [2.8](https://stepik.org/lesson/26559/step/5?unit=8406)
# Продолжим нашу работу в службе безопасности! Разобравшись с тем, какие предикторы могут помогать нам предсказывать запрещенный багаж, давайте применим наши знания для повышения безопасности в аэропорту. Обучим наш алгоритм различать запрещенный и разрешенный багаж на уже имеющихся данных и применим его для сканирования нового багажа!
# 
# Напишите функцию, которая принимает на вход два набора данных. Первый dataframe, как и в предыдущей задаче, содержит информацию об уже осмотренном багаже (запрещенный или нет, вес, длина, ширина, тип сумки). 
# 
# Второй набор данных — это информация о новом багаже, который сканируется прямо сейчас. В данных также есть информация:  вес, длина, ширина, тип сумки и имя пассажира (смотри описание переменных в примере). 
# 
# Используя первый набор данных, обучите регрессионную модель различать запрещенный и разрешенный багаж. При помощи полученной модели для каждого наблюдения в новых данных предскажите вероятность того, что багаж является запрещенным. Пассажиров, чей багаж получил максимальное значение вероятности, мы попросим пройти дополнительную проверку. 
# 
# Итого, ваша функция принимает два набора данных и возвращает имя пассажира с наиболее подозрительным багажом. Если несколько пассажиров получили максимальное значение вероятности, то верните вектор с несколькими именами. 
# 
# В этой задаче для предсказания будем использовать все предикторы, даже если некоторые из них оказались незначимыми. Для предсказания стройте модель без взаимодействия предикторов.

# In[24]:


# код выполняется перед стартом задачи


# In[100]:


import pandas as pd
import numpy as np
from random import randint, choices, uniform, normalvariate, seed
import statsmodels.api as sm
from itertools import combinations 
import numpy as np
import sys
from numpy import mean, array, exp
import names
###################################################################################
#############Формируем train и test датасеты #############
###################################################################################
# Найдем все возможные варианты:
s = 'Weight Height Width Type'.split() 
comb = []
for y in list([x for z in range(1, len(s) + 1) for x in combinations(s,z)]):
    comb.append(y)
comb = [''.join(map(str, x)) for x in comb]
comb.insert(0, 'No_sense')
res = choices(comb)
size = 1000
name = [names.get_first_name() for _ in range(size)]
def create_dataframe(): 
    # Сформируем датасет в зависимости от исхода:
    weight = [normalvariate(23, 3) for _ in range(size)] # 23 кг. - максимальный вес багажа в Аэрофлоте. Остальные параметры тоже по Аэрофлоту.     
    height = [normalvariate(55, 10) for _ in range(size)]
    width = [normalvariate(40, 10) for _ in range(size)]
    type = choices(['сумка', 'чемодан'], k = size) # Просто рэндом.
    train_data = pd.DataFrame(list(zip(weight, height, width, type)), columns = ['weight', 'height', 'width', 'type'])
    train_data.type = pd.get_dummies(train_data.type, drop_first= True).astype('int')
    is_prohibited = [] # Независимую переменную генерируем в зависимости от исхода
    ################################################## Генерация is_prohibited
    if res[0] == comb[0]: # Нет значимых предикторов
        #seed(1)
        z = 1
        pr = 1/(1 + math.exp(-z))
        is_prohibited.extend(choices(['Yes', 'No'],  weights = [1 - pr, pr], k = size))
    elif res[0] == comb[1]: # Weight
        #seed(1)
        z = 1 - 2 * array(weight)/ mean(weight)
        pr = 1/(1+exp(-z))
        for i in range(len(pr)):
            is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
    elif res[0] == comb[2]: # Height
        #seed(1)
        z = 1 - 2 * array(height)/ mean(height)
        pr = 1/(1+exp(-z))
        for i in range(len(pr)):
            is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
    elif res[0] == comb[3]: # Width
        #seed(1)
        z = 1 - 2 * array(width)/ mean(width)
        pr = 1/(1+exp(-z))
        for i in range(len(pr)):
            is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
    elif res[0] == comb[4]: # Type
        #seed(1)
        z = 1 - 2 * array(train_data.type )/ mean(train_data.type )
        pr = 1/(1+exp(-z))
        for i in range(len(pr)):
            is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
    elif res[0] == comb[5]: # WeightHeight
        #seed(1)
        z = 1 - 2 * array(weight)/ mean(weight) + 2 * array(height)/ mean(height)
        pr = 1/(1+exp(-z))
        for i in range(len(pr)):
            is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
    elif res[0] == comb[6]: # WeightWidth
        #seed(1)
        z = 1 - 2.1 * array(weight)/ mean(weight) - 3 * array(width)/ mean(width)
        pr = 1/(1+exp(-z))
        for i in range(len(pr)):
            is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
    elif res[0] == comb[7]: # WeightType
        #seed(1)
        z = 1 - 2 * array(weight)/ mean(weight) + array(train_data.type)/ mean(train_data.type)
        pr = 1/(1+exp(-z))
        for i in range(len(pr)):
            is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
    elif res[0] == comb[8]: # HeightWidth
        #seed(1)
        z = 1 - 2 * array(height)/ mean(height) + 3* array(width)/ mean(width)
        pr = 1/(1+exp(-z))
        for i in range(len(pr)):
            is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
    elif res[0] == comb[9]: # HeightType
        #seed(1)
        z = 1 - 2 * array(height)/ mean(height) + array(train_data.type)/ mean(train_data.type)
        pr = 1/(1+exp(-z))
        for i in range(len(pr)):
            is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
    elif res[0] == comb[10]: # WidthType
        #seed(1)
        z = 1 - 2 * array(width)/ mean(width) + array(train_data.type)/ mean(train_data.type)
        pr = 1/(1+exp(-z))
        for i in range(len(pr)):
            is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
    elif res[0] == comb[11]: # WeightHeightWidth
        #seed(1)
        z = 1 - 2 * array(weight)/ mean(weight) + 3 * array(height)/ mean(height) - 4 * array(width)/ mean(width)
        pr = 1/(1+exp(-z))
        for i in range(len(pr)):
            is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
    elif res[0] == comb[12]: # WeightHeightType
        #seed(1)
        z = 1 - 1.1*array(weight)/ mean(weight) - 1.5 * array(height)/ mean(height) + array(train_data.type)/ mean(train_data.type)
        pr = 1/(1+exp(-z))
        for i in range(len(pr)):
            is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
    elif res[0] == comb[13]: # WeightWidthType
        #seed(1)
        z = 1 - 2.5 * array(weight)/ mean(weight) - 4 * array(width)/ mean(width) + array(train_data.type)/ mean(train_data.type)
        pr = 1/(1+exp(-z))
        for i in range(len(pr)):
            is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
    elif res[0] == comb[14]: # HeightWidthType
        #seed(1)
        z = 1 - 3 * array(height)/ mean(height) - 3 * array(width)/ mean(width) + array(train_data.type)/ mean(train_data.type)
        pr = 1/(1+exp(-z))
        for i in range(len(pr)):
            is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
    elif res[0] == comb[15]: # WeightHeightWidthType
        #seed(1)
        z = 1 - 3 * array(weight)/ mean(weight) + 3 * array(height)/ mean(height) - 4 * array(width)/ mean(width) + array(train_data.type)/ mean(train_data.type)
        pr = 1/(1+exp(-z))
        for i in range(len(pr)):
            is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
##################################################################################################
    df = pd.DataFrame(list(zip(is_prohibited, weight, height, width, train_data.type)), columns = ['is_prohibited', 'weight', 'height', 'width', 'type'])
    df.is_prohibited = df.is_prohibited.map(dict(Yes=1, No=0)) # Финальный датасет
    return df
train_data = create_dataframe()
test_data = create_dataframe().drop('is_prohibited', 1)
test_data['name'] = name


# In[101]:


# студент вводит свой код


# In[102]:


def most_suspicious(train_data, test_data):
    return ['Mike']


# In[103]:


# код, который выполняется после отправки решения


# In[104]:


defined_vars = [var for var in locals()]

if 'most_suspicious' not in defined_vars:
    sys.exit("\nВы должны создать функцию с именем most_suspicious.")
feedback = '\nПроверьте ваши вычисления.\nВаш ответ {student_answer}\nПравильный ответ {master_answer}'
    
def get_master_solution(train_data, test_data):  
    model = sm.GLM.from_formula('is_prohibited ~ weight + height + width + type', family = sm.families.Binomial(), data = train_data)
    result = model.fit()
    p = array(result.predict(test_data))
    most = np.argwhere(p == np.amax(p)).flatten().tolist() # Список самых подозрительных пассажиров
    rslt = test_data.name[most].tolist()
    return rslt

master_answer = get_master_solution(train_data, test_data)
student_answer = most_suspicious(train_data, test_data)
if not set(master_answer) == set(student_answer) :
            sys.exit(feedback.format(student_answer=student_answer, master_answer=master_answer))

print('correct')

