# код выполняется перед стартом задачи

import pandas as pd
import numpy as np
import sys
from sklearn.datasets import load_iris

iris = load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data']],
                     columns= iris['feature_names'])

iris_master = iris_df.copy()

# студент вводит свой код
iris_df  # Датафрейм с данными iris
iris_df['important_cases'] = 1

# код, который выполняется после отправки решения

defined_vars = [var for var in locals()]


if 'iris_df' not in defined_vars:
    sys.exit("\nВы должны создать датафрейм с именем iris_df.")


if 'important_cases' not in iris_df.columns:
    sys.exit("\nВы должны создать столбец с именем important_cases.")


feedback = '\nПроверьте ваши вычисления.\nВаш ответ \n{student_answer} \
            \nПравильный ответ \n{master_answer}'


def is_equal(x,y): 
    return x == y    


def get_important_cases(test_data):
    return (test_data > test_data.mean(axis=0)).sum(axis=1).apply(lambda x:
                                                                  "Yes" if x >= 3 
                                                                        else "No")

for i in range(1):

    test_data = iris_master
    master_answer = get_important_cases(test_data)
    student_answer = iris_df['important_cases']


    for i, j in zip(master_answer, student_answer):
        if not is_equal(i, j):
            sys.exit(feedback.format(student_answer=student_answer, master_answer=master_answer))

print('correct')
