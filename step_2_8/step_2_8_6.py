# код выполняется перед стартом задачи

import pandas as pd
import numpy as np
import sys
from scipy.stats import shapiro
from random import randint


# студент вводит свой код

def normality_test(dataset):
    
    return [1, 1, 1]

# код, который выполняется после отправки решения

defined_vars = [var for var in locals()]

if 'normality_test' not in defined_vars:
    sys.exit("\nВы должны создать функцию с именем normality_test.")


feedback = '\nПроверьте ваши вычисления.\nВаш ответ {student_answer}\nПравильный ответ {master_answer}'
    
def is_almost_equal(x,y, epsilon=10**(-2)): 
    return abs(float(x)-float(y)) <= epsilon     
    
def get_master_solution(test_data):
    result = test_data.select_dtypes('number').apply(shapiro).loc[1].to_list()
    return result    


# test from stepic

test_data = pd.read_csv('https://stepic.org/media/attachments/course/524/test.csv')

master_answer = get_master_solution(test_data)
student_answer = normality_test(test_data)
    
for i, j in zip(master_answer, student_answer):
    if not is_almost_equal(i, j):
        sys.exit(feedback.format(student_answer=student_answer, master_answer=master_answer))


# random test
            
for i in range(10):
    col_num = randint(6, 10)
    test_data = pd.DataFrame({})
    for i in range(col_num):
        flip = randint(0, 3)
        is_int = flip == 0
        is_float = flip == 1
        is_object = flip == 2
        is_bool = flip == 3

        if is_int:
            X = np.random.binomial(10, 0.5, 100)
        elif is_float:
            X = np.random.random(100)
        elif is_object:
            X = ['string']*100
        else:
            X = True
        test_data[f'col_{i}'] = X
    
    
    
    master_answer = get_master_solution(test_data)
    student_answer = normality_test(test_data)
    
    for i, j in zip(master_answer, student_answer):
        if not is_almost_equal(i, j):
            sys.exit(feedback.format(student_answer=student_answer, master_answer=master_answer))

print('correct')
