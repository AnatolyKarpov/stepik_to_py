#!/usr/bin/env python
# coding: utf-8

# https://stepik.org/lesson/26559/step/3?unit=8406
# 
"""
Если в нашей модели есть количественные предикторы, то в интерцепте мы будем иметь значение, соответствующее базовому уровню категориальных предикторов и нулевому уровню количественных. Это не всегда осмысленно. Например, нам не интересен прогноз для людей нулевого возраста или роста. В таких ситуациях количественную переменную имеет смысл предварительно центрировать так, чтобы ноль являлся средним значением переменной. Самый простой способ центрировать переменную — отнять от каждого наблюдения среднее значение всех наблюдений.
 
$$xcentered_{i}= x_{i} - \bar{x} $$

В этом задании вашей задачей будет  написать функцию centered, которая получает на вход датафрейм и имена переменных, которые необходимо центрировать так, как это описано выше. Функция должна возвращать этот же датафрейм, только с центрированными указанными переменными.
"""

# код выполняется перед стартом задачи

import numpy as np
import pandas as pd
import sys

# студент вводит свой код

def centered(test_data, var_names):
    return test_data


# код, который выполняется после отправки решения

defined_vars = [var for var in locals()]

if 'centered' not in defined_vars:
    sys.exit("\nВы должны создать функцию с именем centered.")


feedback = '\nПроверьте ваши вычисления.\nВаш ответ \n{student_answer}\nПравильный ответ \n{master_answer}'

def is_df_almost_equal(df_1, df_2, epsilon=10**(-2)):
    if type(df_1) is pd.DataFrame and type(df_2) is pd.DataFrame and df_1.shape == df_2.shape:
        return (abs(df_1 - df_2) <= epsilon).all(axis=None)
    return False

def get_master_solution(test_data, var_names):
    result = test_data.copy()
    result[var_names] = result[var_names] - result[var_names].mean()
    return result

for i in range(10):
    columns = np.array(["X1", "X2", "X3", "X4"])
    
    mu = np.random.randint(5, 10, 4)
    sigma = 2
    data = np.random.normal(mu, sigma, (5, 4)).round(1)    
    test_data = pd.DataFrame(data, columns = columns)
    
    vars_flag = np.random.choice([True,False],4)
    var_names = columns[vars_flag]
    
    master_answer = get_master_solution(test_data,var_names)
    student_answer = centered(test_data,var_names)
    
    if not is_df_almost_equal(master_answer, student_answer):
        sys.exit(feedback.format(student_answer=student_answer, master_answer=master_answer))

print('correct')

