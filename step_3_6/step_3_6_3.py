"""
https://stepik.org/lesson/26672/step/3?unit=8484

Напишите функцию get_pc, которая получает на вход DataFrame с произвольным числом количественных переменных.
Функция должна выполнять анализ главных компонент и возвращать две колонки со значениями первой и второй главной компоненты.
Новые переменные должны называться "PC1"  и "PC2" соответственно.

Пример работы функции:
```python
>>> test_data = pd.read_csv("https://stepic.org/media/attachments/course/524/pca_test.csv")

>>> test_data
  V1 V2 V3 V4 V5
1 13 15 12 13 12
2 16 11  8 12  6
3 15  7 10 12 13
4 12 11  6  6  4
5 11 13 13 10 12

>>> get_pc(test_data)
       PCA1      PCA2
0 -4.500822 -2.364595
1  3.039182  1.929687
2 -2.752467  5.076265
3  7.837194 -1.688535
4 -3.623088 -2.952822
```
 
Для выполнения анализа главных компонент используйте функцию [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
"""

import sys
from typing import Callable, List, Tuple

import pandas as pd
from sklearn.decomposition import PCA


class WrongAnswer(Exception):
    def __init__(self, err_msg):
        msg = (
            '\nПроверьте ваши вычисления.'
            '\nDataFrame, который возвращает ваша функция не проходит проверку:'
            '\n{err_msg}'
        )
        super().__init__(msg.format(err_msg=err_msg))


def result_is_correct(actual: pd.DataFrame, expected: pd.DataFrame):
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_less_precise=1
        )
    except AssertionError as e:
        raise WrongAnswer(e)


def get_test_cases() -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    test_cases = []

    # 1
    test_columns = ['V1', 'V2', 'V3', 'V4', 'V5']
    test_data = [
        [13, 15, 12, 13, 12],
        [16, 11, 8, 12, 6],
        [15, 7, 10, 12, 13],
        [12, 11, 6, 6, 4],
        [11, 13, 13, 10, 12]
    ]

    result_columns = ['PCA1', 'PCA2']
    result_data = [
        [-4.501, -2.365],
        [3.039, 1.93],
        [-2.752, 5.076],
        [7.837, -1.689],
        [-3.623, -2.953]
    ]

    test_cases.append(
        (
            pd.DataFrame(data=test_data, columns=test_columns),
            pd.DataFrame(data=result_data, columns=result_columns)
        )
    )

    # 2
    test_columns = ['V1', 'V2', 'V3', 'V4']
    test_data = [
        [18, 9, 11, 15],
        [16, 14, 12, 9],
        [20, 15, 12, 15],
        [2, 18, 4, 7],
        [1, 4, 14, 8],
        [13, 5, 8, 14]
    ]

    result_columns = ['PCA1', 'PCA2']
    result_data = [
        [-7.128, -2.32],
        [-3.631, 2.866],
        [-9.524, 2.939],
        [10.759, 9.057],
        [11.014, -7.539],
        [-1.489, -5.002]
    ]

    test_cases.append(
        (
            pd.DataFrame(data=test_data, columns=test_columns),
            pd.DataFrame(data=result_data, columns=result_columns)
        )
    )

    # 3
    test_columns = ['V1', 'V2']
    test_data = [
        [-1, -1],
        [-2, -1],
        [-3, -2],
        [1, 1],
        [2, 1],
        [3, 2]
    ]

    result_columns = ['PCA1', 'PCA2']
    result_data = [
        [1.383, 0.294],
        [2.222, -0.251],
        [3.605, 0.042],
        [-1.383, -0.294],
        [-2.222, 0.251],
        [-3.605, -0.042]
    ]

    test_cases.append(
        (
            pd.DataFrame(data=test_data, columns=test_columns),
            pd.DataFrame(data=result_data, columns=result_columns)
        )
    )

    return test_cases


# студент вводит свой код

def get_pc(x: pd.DataFrame) -> pd.DataFrame:
    return x

# код, который выполняется после отправки решения


def check_student_func(student_func: Callable[[pd.DataFrame], pd.DataFrame]):
    for input_data, expected_answer in get_test_cases():
        student_answer = student_func(input_data)
        result_is_correct(student_answer, expected_answer)


def master_solution(x: pd.DataFrame) -> pd.DataFrame:
    components_names = ['PCA1', 'PCA2']
    pca = PCA(n_components=len(components_names))
    pca_components = pca.fit_transform(x.values)
    return pd.DataFrame(data=pca_components, columns=components_names)


def main():
    try:
        check_student_func(get_pc)
    except Exception as e:
        sys.exit(e)
    else:
        print('correct')


if __name__ == '__main__':
    main()
