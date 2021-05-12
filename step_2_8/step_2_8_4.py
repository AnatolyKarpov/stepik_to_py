#!/usr/bin/env python
# coding: utf-8

# [2.8](https://stepik.org/lesson/26559/step/4?unit=8406)
# Представьте, что мы работаем в аэропорту в службе безопасности и сканируем багаж пассажиров. В нашем распоряжении есть информация о результатах проверки багажа за предыдущие месяцы. Про каждую вещь мы знаем:
# 
# * являлся ли багаж запрещенным - is_prohibited (No - разрешенный, Yes - запрещенный) 
# * его массу (кг) - weight
# * длину (см) - length
# * ширину (см) - width
# * тип багажа (сумка или чемодан) - type.
# 
# Напишите функцию **get_features** , которая получает на вход набор данных о багаже. Строит логистическую регрессию, где зависимая переменная - являлся ли багаж запрещенным, а предикторы - остальные переменные, и возвращает вектор с названиями статистически значимых переменных (p < 0.05) (в модели без взаимодействия). Если в данных нет значимых предикторов, функция возвращает строку с сообщением  "Prediction makes no sense".

# In[519]:


# код выполняется перед стартом задачи


# In[525]:


import pandas as pd
import numpy as np
from random import randint, choices, uniform, normalvariate, seed
import statsmodels.api as sm
from itertools import combinations 
import numpy as np
import sys
from numpy import mean, array, exp
###################################################################################
#############Делаем так, чтобы ответы выпадали более-менее равномерно#############
###################################################################################
# Найдем все возможные варианты:
s = 'Weight Height Width Type'.split() 
comb = []
for y in list([x for z in range(1, len(s) + 1) for x in combinations(s,z)]):
    comb.append(y)
comb = [''.join(map(str, x)) for x in comb]
comb.insert(0, 'No_sense')
res = choices(comb) # Определим исход
size = 1000
# Сформируем датасет в зависимости от исхода:
seed(1)
weight = [normalvariate(23, 3) for _ in range(size)] # 23 кг. - максимальный вес багажа в Аэрофлоте. Остальные параметры тоже по Аэрофлоту.     
height = [normalvariate(55, 10) for _ in range(size)]
width = [normalvariate(40, 10) for _ in range(size)]
type = choices(['сумка', 'чемодан'], k = size) # Просто рэндом.
train_data = pd.DataFrame(list(zip(weight, height, width, type)), columns = ['weight', 'height', 'width', 'type'])
train_data.type = pd.get_dummies(train_data.type, drop_first= True).astype('int')
is_prohibited = [] # Независимую переменную генерируем в зависимости от исхода
################################################## Генерация is_prohibited
if res[0] == comb[0]: # Нет значимых предикторов
    seed(1)
    z = 1
    pr = 1/(1 + math.exp(-z))
    is_prohibited.extend(choices(['Yes', 'No'],  weights = [1 - pr, pr], k = size))
elif res[0] == comb[1]: # Weight
    seed(1)
    z = 1 - 2 * array(weight)/ mean(weight)
    pr = 1/(1+exp(-z))
    for i in range(len(pr)):
        is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
elif res[0] == comb[2]: # Height
    seed(1)
    z = 1 - 2 * array(height)/ mean(height)
    pr = 1/(1+exp(-z))
    for i in range(len(pr)):
        is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
elif res[0] == comb[3]: # Width
    seed(1)
    z = 1 - 2 * array(width)/ mean(width)
    pr = 1/(1+exp(-z))
    for i in range(len(pr)):
        is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
elif res[0] == comb[4]: # Type
    seed(1)
    z = 1 - 2 * array(train_data.type )/ mean(train_data.type )
    pr = 1/(1+exp(-z))
    for i in range(len(pr)):
        is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
elif res[0] == comb[5]: # WeightHeight
    seed(1)
    z = 1 - 2 * array(weight)/ mean(weight) + 2 * array(height)/ mean(height)
    pr = 1/(1+exp(-z))
    for i in range(len(pr)):
        is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
elif res[0] == comb[6]: # WeightWidth
    seed(1)
    z = 1 - 2.1 * array(weight)/ mean(weight) - 3 * array(width)/ mean(width)
    pr = 1/(1+exp(-z))
    for i in range(len(pr)):
        is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
elif res[0] == comb[7]: # WeightType
    seed(1)
    z = 1 - 2 * array(weight)/ mean(weight) + array(train_data.type)/ mean(train_data.type)
    pr = 1/(1+exp(-z))
    for i in range(len(pr)):
        is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
elif res[0] == comb[8]: # HeightWidth
    seed(1)
    z = 1 - 2 * array(height)/ mean(height) + 3* array(width)/ mean(width)
    pr = 1/(1+exp(-z))
    for i in range(len(pr)):
        is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
elif res[0] == comb[9]: # HeightType
    seed(1)
    z = 1 - 2 * array(height)/ mean(height) + array(train_data.type)/ mean(train_data.type)
    pr = 1/(1+exp(-z))
    for i in range(len(pr)):
        is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
elif res[0] == comb[10]: # WidthType
    seed(1)
    z = 1 - 2 * array(width)/ mean(width) + array(train_data.type)/ mean(train_data.type)
    pr = 1/(1+exp(-z))
    for i in range(len(pr)):
        is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
elif res[0] == comb[11]: # WeightHeightWidth
    seed(1)
    z = 1 - 2 * array(weight)/ mean(weight) + 3 * array(height)/ mean(height) - 4 * array(width)/ mean(width)
    pr = 1/(1+exp(-z))
    for i in range(len(pr)):
        is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
elif res[0] == comb[12]: # WeightHeightType
    seed(1)
    z = 1 - 1.1*array(weight)/ mean(weight) - 1.5 * array(height)/ mean(height) + array(train_data.type)/ mean(train_data.type)
    pr = 1/(1+exp(-z))
    for i in range(len(pr)):
        is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
elif res[0] == comb[13]: # WeightWidthType
    seed(1)
    z = 1 - 2.5 * array(weight)/ mean(weight) - 4 * array(width)/ mean(width) + array(train_data.type)/ mean(train_data.type)
    pr = 1/(1+exp(-z))
    for i in range(len(pr)):
        is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
elif res[0] == comb[14]: # HeightWidthType
    seed(1)
    z = 1 - 3 * array(height)/ mean(height) - 3 * array(width)/ mean(width) + array(train_data.type)/ mean(train_data.type)
    pr = 1/(1+exp(-z))
    for i in range(len(pr)):
        is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
elif res[0] == comb[15]: # WeightHeightWidthType
    seed(1)
    z = 1 - 3 * array(weight)/ mean(weight) + 3 * array(height)/ mean(height) - 4 * array(width)/ mean(width) + array(train_data.type)/ mean(train_data.type)
    pr = 1/(1+exp(-z))
    for i in range(len(pr)):
        is_prohibited.extend(choices(['Yes', 'No'],weights = [1 - pr[i], pr[i]]))
##################################################################################################
train_data = pd.DataFrame(list(zip(is_prohibited, weight, height, width, train_data.type)), columns = ['is_prohibited', 'weight', 'height', 'width', 'type'])
train_data.is_prohibited = train_data.is_prohibited.map(dict(Yes=1, No=0)) # Финальный датасет


# In[526]:


# студент вводит свой код


# In[527]:


def get_features(train_data):
    return ['type', 'weight']


# In[528]:


# код, который выполняется после отправки решения


# In[529]:


defined_vars = [var for var in locals()]

if 'get_features' not in defined_vars:
    sys.exit("\nВы должны создать функцию с именем get_features.")
feedback = '\nПроверьте ваши вычисления.\nВаш ответ {student_answer}\nПравильный ответ {master_answer}'
    
def get_master_solution(train_data):  
    model = sm.GLM.from_formula('is_prohibited ~ weight + height + width + type', family = sm.families.Binomial(), data = train_data)
    result = model.fit()
    sum = 0
    features = []
    for i in range(1, len(result.pvalues)):
        if result.pvalues[i] < 0.05:
            features.append(result.pvalues.index[i])
            sum +=1
    if sum == 0:
        rslt = 'Prediction makes no sense'
    else:
        rslt = features
    return rslt

master_answer = get_master_solution(train_data)
student_answer = get_features(train_data)
if not set(master_answer) == set(student_answer) :
            sys.exit(feedback.format(student_answer=student_answer, master_answer=master_answer))

print('correct')

