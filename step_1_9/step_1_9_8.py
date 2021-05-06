"""
https://stepik.org/lesson/26186/step/8?unit=8128

Доктор Пилюлькин решил вооружиться статистикой, чтобы сравнить эффективность трех лекарств! Давайте поможем ему и напишем функцию max_resid, которая получает на вход две pd.Series: тип лекарства и результат его применения. 

Drugs - фактор с тремя градациями: drug_1, drug_2, drug_3.     
Result - фактор с двумя градациями: positive, negative.

Функция должна строить таблицу сопряженности, а затем находить ячейку с максимальным  значением стандартизированного остатка и возвращать кортеж из двух элементов: название строчки и столбца этой ячейки.

Вам могут понадобиться следующие функции:    
[pandas.crosstab](https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html)    
[pandas.Series.argmax](https://pandas.pydata.org/docs/reference/api/pandas.Series.argmax.html)    
[statsmodels.stats.contingency_tables.Table](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.Table.html)    

Изучите справку по этим функциям

Пример работы функции на одном из вариантов:
```python
>>> df = pd.read_csv('data.csv')
>>> df.head()
    Drugs   Results
0  drug_2  positive
1  drug_1  negative
2  drug_3  positive
3  drug_2  positive
4  drug_2  negative

>>> max_resid(rows=df['Drugs'], columns=df['Results'])
('drug_1', 'positive')
```
"""

import sys

import pandas as pd
import statsmodels.api as sm

from itertools import repeat
from typing import Callable, List, Tuple


class WrongAnswer(Exception):
    def __init__(self, actual: Tuple[str, str], expected: Tuple[str, str]):
        msg = (
            '\nПроверьте ваши вычисления.'
            '\nВаш ответ {actual}'
            '\nПравильный ответ {expected}'
        )

        super().__init__(
            msg.format(
                actual=actual,
                expected=expected
            )
        )


# студент вводит свой код

def max_resid(rows: pd.Series, columns: pd.Series) -> Tuple[str, str]:
    """
    Функция должна строить таблицу сопряженности, 
    а затем находить ячейку с максимальным значением 
    стандартизированного остатка и возвращать кортеж из двух элементов: 
    название строчки и столбца этой ячейки
    
    Parameters
    ----------
    rows: pd.Series
    columns: pd.Series
    Количество элементов в строках и колонках совпадает
    
    Returns
    -------
    Возвращает кортеж из двух элементов: 
    название строчки и стобца ячейки 
    с максимальным значением стандартизированного остатка
        Tuple[str, str]
    """
    # Напишите ваш код здесь
    return max_row_name, max_column_name


# код, который выполняется после отправки решения

def result_is_correct(actual: Tuple[str, str], expected: Tuple[str, str]) -> bool:
    return actual == expected


def get_test_cases() -> List[Tuple[int, pd.DataFrame, Tuple[str, str]]]:
    test_cases = [
        (
            1,
            [
                *tuple(repeat(('drug_1', 'positive'), 2)),
                *tuple(repeat(('drug_1', 'negative'), 2)),
                *tuple(repeat(('drug_2', 'positive'), 3)),  # <-
                *tuple(repeat(('drug_2', 'negative'), 2)),
                *tuple(repeat(('drug_3', 'positive'), 2)),
                *tuple(repeat(('drug_3', 'negative'), 2)),
            ],
            ('drug_2', 'positive')
        ),
        (
            2,
            [
                *tuple(repeat(('drug_1', 'positive'), 1)),
                *tuple(repeat(('drug_1', 'negative'), 20)),  # <-
                *tuple(repeat(('drug_2', 'positive'), 15)),
                *tuple(repeat(('drug_2', 'negative'), 45)),
                *tuple(repeat(('drug_3', 'positive'), 12)),
                *tuple(repeat(('drug_3', 'negative'), 36)),
            ],
            ('drug_1', 'negative')
        ),
        (
            3,
            [
                *tuple(repeat(('drug_1', 'positive'), 10)),
                *tuple(repeat(('drug_1', 'negative'), 10)),
                *tuple(repeat(('drug_2', 'positive'), 10)),
                *tuple(repeat(('drug_2', 'negative'), 10)),
                *tuple(repeat(('drug_3', 'positive'), 11)),  # <-
                *tuple(repeat(('drug_3', 'negative'), 10)),
            ],
            ('drug_3', 'positive')
        ),

    ]

    columns = ['Drugs', 'Results']
    test_cases = [
        (
            idx,
            pd.DataFrame(data=data, columns=columns),
            answer
        )
        for idx, data, answer in test_cases
    ]

    return test_cases


def check_student_func(student_func: Callable[[pd.Series, pd.Series], Tuple[str, str]]):
    for idx, input_data, expected_answer in get_test_cases():
        student_answer = student_func(input_data['Drugs'], input_data['Results'])

        if not result_is_correct(student_answer, expected_answer):
            raise WrongAnswer(actual=student_answer, expected=expected_answer)


def master_solution(rows: pd.Series, columns: pd.Series) -> Tuple[str, str]:
    """
    Функция должна строить таблицу сопряженности, 
    а затем находить ячейку с максимальным значением 
    стандартизированного остатка и возвращать кортеж из двух элементов: 
    название строчки и столбца этой ячейки
    
    Parameters
    ----------
    rows: pd.Series
    columns: pd.Series
    Количество элементов в строках и колонках совпадает
    
    Returns
    -------
    Возвращает кортеж из двух элементов: 
    название строчки и стобца ячейки 
    с максимальным значением стандартизированного остатка
        Tuple[str, str]
    """

    cross_tab = pd.crosstab(index=rows, columns=columns)
    table = sm.stats.Table(cross_tab)

    standardized_residuals = table.standardized_resids

    max_row_name = standardized_residuals.max(axis=1).idxmax()
    max_column_name = standardized_residuals.max(axis=0).idxmax()

    return max_row_name, max_column_name


def main():
    try:
        check_student_func(max_resid)
    except Exception as e:
        sys.exit(e)
    else:
        print('correct')


if __name__ == '__main__':
    main()
