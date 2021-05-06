# код выполняется перед стартом задачи

import pandas as pd
import numpy as np
import sys
from scipy.stats import chisquare

# студент вводит свой код


def most_significant(test_data):
    return [1]
    

# код, который выполняется после отправки решения


defined_vars = [var for var in locals()]

if 'most_significant' not in defined_vars:
    sys.exit("\nВы должны создать функцию с именем most_significant.")


feedback = '\nПроверьте ваши вычисления.\nВаш ответ {student_answer}\nПравильный ответ {master_answer}'


def is_equal(x,y): 
    x.sort()
    y.sort()
    return set(x) == set(y)     


def get_master_solution(test_data):
    results = [[chisquare(test_data[sequence].value_counts())[1], sequence] 
               for sequence in test_data.columns]
    results = np.array(results)
    min_p = np.min(results[:,0].astype(float))
    return results[np.where(results[:,0].astype(float) == min_p)][:, 1].flatten()


for i in range(5):
    if i == 0 :
        test_data = pd.read_csv("https://stepic.org/media/attachments/course/524/test_data.csv")
    if i == 1:
        test_data = pd.DataFrame({"V1": list("ATGCATGCCGTAC"),
                   "V2": list("ATGCCTGCCGTCC"),
                   "V3": list("ATGCCCGCCGCCC")},
                  dtype="category")
    if i == 2:
        test_data = pd.DataFrame({"V1": list("ATGCATGCCGTAC"),
                   "V2": list("ATGCCTGCCGTCC"),
                   "V3": list("ATGCCCGCCGCCC"),
                   "V4": list("ATGCCCGCCGCCC")},
                  dtype="category")
    if i == 3:
        test_data = pd.DataFrame({"V1": list("ATGCATGCCGTAC"),
                   "V2": list("ATGCCTGCCGTCC"),
                   "V3": list("AAAAAAAAAAACC")},
                  dtype="category")
    if i == 4:
        test_data = pd.DataFrame({"V1": list("AAAAAAAAAAACC"),
                   "V2": list("ATGCCTGCCGTCC"),
                   "V3": list("ATGCCCGCCGCCC")},
                  dtype="category")

    master_answer = get_master_solution(test_data)
    student_answer = most_significant(test_data)

    
    if not is_equal(master_answer, student_answer):
        sys.exit(feedback.format(student_answer=student_answer, master_answer=master_answer))

print('correct')
