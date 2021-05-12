#!/usr/bin/env python
# coding: utf-8

# https://stepik.org/lesson/26559/step/8?unit=8406
# 
"""
Напишите функцию normality_by, которая принимает на вход dataframe c тремя переменными. Первая переменная количественная, вторая и третья имеют две градации и разбивают наши наблюдения на группы. Функция должна проверять распределение на нормальность в каждой получившейся группе и возвращать dataframe с результатами применения теста shapiro.test (формат вывода смотри ниже).
 
Итого: функция должна возвращать dataframe размером 4 на 3. 

Название столбцов:   
1 — имя первой группирующей переменной  
2 — имя второй группирующей переменной  
3 — p_value   

Подсказка: хороший пример того, когда при помощи функции group_by и summarise из пакета dplyr решается в три строчки! Кстати, обработка данных при помощи dplyr разбирается в курсе Основы программирования на R, прикладываю ссылку на урок.
"""

# код выполняется перед стартом задачи

import pandas as pd
import numpy as np
import sys
from scipy.stats import shapiro


# студент вводит свой код
def normality_by(test_data):
    return pd.DataFrame(np.ones((4,3)),  columns = ['y', 'z', 'p_value'])


# код, который выполняется после отправки решения

defined_vars = [var for var in locals()]

if 'normality_by' not in defined_vars:
    sys.exit("\nВы должны создать функцию с именем normality_by.")
    
feedback = '\nПроверьте ваши вычисления.\nВаш ответ \n{student_answer}\nПравильный ответ \n{master_answer}'

def is_df_almost_equal(df_1, df_2, epsilon=10**(-2)):
    if type(df_1) is pd.DataFrame and type(df_2) is pd.DataFrame and df_1.shape == df_2.shape:
        return (abs(df_1 - df_2) <= epsilon).all(axis=None)
    return False

def get_master_solution(test_data):
    rows = []
    for (y,z), group in test_data.groupby(['y','z']):
        rows.append([y, z, shapiro(group.x).pvalue])
    result = pd.DataFrame(rows, columns = ['y', 'z', 'p_value'])
    return result

for i in range(10):
    n = 40
    x = np.random.normal(10, 2, n).round(1)  
    y = np.random.choice([0,1], n)
    z = np.random.choice([2,3], n)
    
    test_data = pd.DataFrame({'x': x, 'y': y, 'z': z})
    
    master_answer = get_master_solution(test_data)
    student_answer = normality_by(test_data)
    
    if not is_df_almost_equal(master_answer, student_answer):
        sys.exit(feedback.format(student_answer=student_answer, master_answer=master_answer))

print('correct')

