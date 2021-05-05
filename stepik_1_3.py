# код выполняется перед стартом задачи

import pandas as pd
import numpy as np
import sys
from scipy.stats import chi2_contingency, fisher_exact
from random import randint


# студент вводит свой код

def smart_test(test_data):
    return [1, 1, 1]

# код, который выполняется после отправки решения

defined_vars = [var for var in locals()]

if 'smart_test' not in defined_vars:
    sys.exit("\nВы должны создать функцию с именем smart_test.")

feedback = '\nПроверьте ваши вычисления.\nВаш ответ {student_answer}\nПравильный ответ {master_answer}'


def is_almost_equal(x, y, epsilon=10 ** (-2)):
    return abs(float(x) - float(y)) <= epsilon


def get_master_solution(test_data):
    data_crosstab = pd.crosstab(test_data['X'],
                                test_data['Y'],
                                margins=False)

    if np.concatenate(data_crosstab.values).min() < 5:
        oddsratio, p = fisher_exact(data_crosstab)
        result = [p]
    else:
        chi2, p, df, expected = chi2_contingency(data_crosstab)
        result = [chi2, df, p]

    return result


for i in range(10):
    flip = randint(0, 1)
    is_fisher_test = flip == 1

    if is_fisher_test:
        X = np.random.binomial(1, 0.5, 10)
        Y = np.random.binomial(1, 0.5, 10)
    else:
        X = np.random.binomial(1, 0.5, 100)
        Y = np.random.binomial(1, 0.5, 100)

    test_data = pd.DataFrame({'X': X, 'Y': Y})

    master_answer = get_master_solution(test_data)
    student_answer = smart_test(test_data)

    for i, j in zip(master_answer, student_answer):
        if not is_almost_equal(i, j):
            sys.exit(feedback.format(student_answer=student_answer, master_answer=master_answer))

print('correct')
