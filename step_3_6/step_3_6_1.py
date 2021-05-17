#!/usr/bin/env python
# coding: utf-8

# [3.6.1](https://stepik.org/lesson/26672/step/1?auth=login&unit=8484) Напишите функцию smart_hclust, которая получает на вход dataframe  с произвольным числом количественных переменных и число кластеров, которое необходимо выделить при помощи иерархической кластеризации.
# Функция должна в исходный набор данных добавлять новую переменную фактор - **cluster**  -- номер кластера, к которому отнесено каждое из наблюдений.

# In[106]:


# код выполняется перед стартом задачи
import sys
import pandas as pd
#import numpy as np
from random import randint, uniform
#import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
#import statsmodels.api as sm
#from statsmodels.formula.api import ols
# Генерируем df:
n_col = randint(3, 20)
size = 1000
column_names = ['V' + str(i + 1) for i in range(n_col)]
df = pd.DataFrame(columns = column_names)
for i in range(len(df.columns)):
    df[column_names[i]] = [uniform(0, 1) for _ in range(size)]
# Генерируем количество кластеров:
n_cluster = randint(2, n_col//2 )


# In[107]:


# студент вводит свой код


# In[108]:


def smart_hclust(df, n_cluster):
    #df1 = pd.DataFrame(columns = column_names)
    #for i in range(len(df1.columns)):
    #    df1[column_names[i]] = [uniform(0, 1) for _ in range(size)]
    #df1['cluster'] = [randint(0,3) for _ in range(size)]
    return df1


# In[109]:


# код, который выполняется после отправки решения


# In[110]:


defined_vars = [var for var in locals()]

if 'smart_hclust' not in defined_vars:
    sys.exit("\nВы должны создать функцию с именем smart_hclust.")

feedback = '\nПроверьте названия и порядок столбцов.\nВаш ответ {student_answer}\nПравильный ответ {master_answer}'
feedback1 = '\nПроверьте ваши вычисления.\nВаш ответ {student_answer}\nПравильный ответ {master_answer}'

def get_master_solution(df, n_cluster):  
    cluster = AgglomerativeClustering(n_clusters = n_cluster, affinity='euclidean', linkage='ward')  
    df['cluster'] =  cluster.fit_predict(df)
    return df

master_answer = get_master_solution(df, n_cluster)
student_answer = smart_hclust(df, n_cluster)

if not list(master_answer.columns) == list(student_answer.columns):
    sys.exit(feedback.format(student_answer=student_answer.columns, master_answer=master_answer.columns))
elif master_answer.compare(student_answer).size != 0:
    sys.exit(feedback1.format(student_answer=student_answer, master_answer=master_answer))

print('correct')

