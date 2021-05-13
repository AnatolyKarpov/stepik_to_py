#!/usr/bin/env python
# coding: utf-8

# https://stepik.org/lesson/26559/step/2?unit=8406
 
"""
Начнем с простого и вспомним, как применять логистическую регрессию в R. Напишите функцию get_coefficients, которая получает на вход dataframe с двумя переменными x ( фактор с произвольным числом градаций) и y ( фактор с двумя градациями ). Функция строит логистическую модель, где y — зависимая переменная, а x — независимая, и возвращает вектор со значением экспоненты коэффициентов модели. 
"""


# код выполняется перед стартом задачи

import numpy as np
import pandas as pd
import sys
import statsmodels.formula.api as sf
from sklearn.linear_model import LogisticRegression
from patsy import dmatrices
from random import randint



# студент вводит свой код

def get_coefficients(test_data):
    return [1, 1, 1]


# код, который выполняется после отправки решения

defined_vars = [var for var in locals()]

if 'get_coefficients' not in defined_vars:
    sys.exit("\nВы должны создать функцию с именем get_coefficients.")
    
feedback = '\nПроверьте ваши вычисления.\nВаш ответ {student_answer}\nПравильный ответ {master_answer}'

def is_almost_equal(x,y, epsilon=10**(-2)): 
    return abs(float(x)-float(y)) <= epsilon  

"""
Решение с использованием библиотеки statsmodels, метод statsmodels.formula.api.logit
Аналогичные методы:
- statsmodels.api.GLM
- statsmodels.formula.api.GLM
"""
def get_master_solution(test_data):
    logit_res = sf.logit('y ~ C(x)', test_data).fit()
    logit_res.summary()
    summary_as_html = logit_res.summary().tables[1].as_html()
    summary_df = pd.read_html(summary_as_html, header=0, index_col=0)[0]
    result = np.array(np.exp(summary_df["coef"]))
    return result

"""
Сравнимые со результаты можно получить с использованием sklearn, 
если сделать регуляризацию неэффективной (C=1e9)
"""
def get_master_solution_sklearn(test_data):
    y, X = dmatrices('y ~ C(x)', test_data,return_type="dataframe")
    sk_model = LogisticRegression(fit_intercept = False, C = 1e9).fit(X, np.ravel(y))
    result = np.exp(sk_model.coef_[0])
    return result

for i in range(10):
    # генерируем число градаций для x
    n = randint(2,4)
    X = np.random.binomial(n, 0.5, 100)+1
    Y = np.random.binomial(1, 0.5, 100)
    test_data = pd.DataFrame({'x':X, 'y':Y})
    
    pivot_tbl = pd.pivot_table(test_data,index="x",columns="y", values="x", aggfunc=len)
    # если в сводной таблице нет значения NaN (иначе не сойдётся)
    if not pivot_tbl.isnull().values.any():
    
        master_answer = get_master_solution(test_data)
        student_answer = get_coefficients(test_data)
    
        for i, j in zip(master_answer, student_answer):
            if not is_almost_equal(i, j):
                sys.exit(feedback.format(student_answer=student_answer, master_answer=master_answer))


print('correct')



