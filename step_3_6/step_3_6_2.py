#!/usr/bin/env python
# coding: utf-8

# [3.6.2](https://stepik.org/lesson/26672/step/2?auth=login&unit=8484) Интересной особенностью кластерного анализа является тот факт, что мы получаем только итоговый ответ, к какому кластеру принадлежит каждое наблюдение. Однако мы не знаем, по каким переменным различаются выделенные кластеры. Поэтому, если нас интересует не только сам факт того, что мы смогли выделить кластеры в наших данных, но мы также хотим понять, чем же они различаются, разумно сравнить кластеры между собой по имеющимся переменным.
# 
# Напишите функцию get_difference, которая получает на вход два аргумента: 
# 
#     df — набор данных с произвольным числом количественных переменных.
#     n_cluster — число кластеров, которое нужно выделить в данных при помощи иерархической кластеризации.
# 
# Функция должна вернуть названия переменных, по которым были обнаружен значимые различия между выделенными кластерами (p < 0.05). Иными словами, после того, как мы выделили заданное число кластеров, мы добавляем в исходные данные новую группирующую переменную — номер кластера, и сравниваем получившиеся группы между собой по количественным переменным при помощи дисперсионного анализа.

# In[22]:


# код выполняется перед стартом задачи
import sys
import pandas as pd
from random import randint, uniform
from sklearn.cluster import AgglomerativeClustering
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Генерируем df:
n_col = randint(3, 20)
size = 1000
column_names = ['V' + str(i + 1) for i in range(n_col)]
df = pd.DataFrame(columns = column_names)
for i in range(len(df.columns)):
    df[column_names[i]] = [uniform(0, 1) for _ in range(size)]
# Генерируем количество кластеров:
n_cluster = randint(2, n_col//2 )


# In[23]:


# студент вводит свой код


# In[24]:


def get_difference(df, n_cluster):
    df1 = df.copy()
    cluster1 = AgglomerativeClustering(n_clusters = n_cluster, affinity='euclidean', linkage='ward')  
    df1['cluster'] =  cluster1.fit_predict(df1)
    df1['cluster'] = pd.factorize(df1.cluster)[0]
    my_formula1 = 'cluster~ V1 + V2' 
    model1 = ols(formula = my_formula1, data=df1).fit()
    anova_table1 = sm.stats.anova_lm(model1, typ=2)
    anova_table1 = anova_table1[anova_table1.index != 'Residual']
    rslt1 =  list(anova_table1[anova_table1['PR(>F)'] < 0.05].index)
    return rslt1


# In[25]:


# код, который выполняется после отправки решения


# In[26]:


defined_vars = [var for var in locals()]

if 'get_difference' not in defined_vars:
    sys.exit("\nВы должны создать функцию с именем get_difference.")

feedback = '\nПроверьте ваши вычисления.\nВаш ответ {student_answer}\nПравильный ответ {master_answer}'

def get_master_solution(df, n_cluster):  
    cluster = AgglomerativeClustering(n_clusters = n_cluster, affinity='euclidean', linkage='ward')  
    df['cluster'] =  cluster.fit_predict(df)
    df['cluster'] = pd.factorize(df.cluster)[0]
    all_columns = "+".join(list(df.columns)[0:-1])
    my_formula = 'cluster~' + all_columns
    model = ols(formula = my_formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table = anova_table[anova_table.index != 'Residual']
    rslt =  list(anova_table[anova_table['PR(>F)'] < 0.05].index)
    return ', '.join(map(str, rslt))

master_answer = get_master_solution(df, n_cluster)
student_answer = get_difference(df, n_cluster)

if not set(student_answer) == set(master_answer):
    sys.exit(feedback.format(student_answer=student_answer, master_answer=master_answer))

print('correct')

