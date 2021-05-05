"""
https://stepik.org/lesson/26186/step/7?unit=8128

Задачка на программирование.

Напишите функцию stat_mode, которая получает на вход список из чисел произвольной длины
и возвращает список с наиболее часто встречаемым значением.
Если наиболее часто встречаемых значений несколько,
функция должна возвращать несколько значений моды в виде списка чисел.
"""

# код выполняется перед стартом задачи

import sys

import pandas as pd

from typing import Callable, List, Tuple


class WrongAnswer(Exception):
    def __init__(self, actual: List[int], expected: List[int]):
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


def result_is_correct(actual: List[int], expected: List[int]) -> bool:
    return set(actual) == set(expected)


def get_test_cases() -> List[Tuple[List[int]]]:
    test_cases = [
        ([1, 1, 2, 2, 2, 3, 3], [2]),
        ([1, 1, 1, 1, 1, 1], [1]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([], []),
        ([1, 2, 2, 2, 3, 3, 4, 5, 5, 5], [2, 5]),
        ([999999, -10000, 8888, -10, 500, -10], [-10]),
    ]
    return test_cases


def check_student_func(student_func: Callable[[List[int]], List[int]]):
    for input_data, expected_answer in get_test_cases():
        student_answer = student_func(input_data)

        if not result_is_correct(student_answer, expected_answer):
            raise WrongAnswer(actual=student_answer, expected=expected_answer)


# студент вводит свой код

def stat_mode(x: List[int]) -> List[int]:
    # Напишите ваш код здесь
    pass

# код, который выполняется после отправки решения


def main():
    try:
        check_student_func(stat_mode)
    except Exception as e:
        sys.exit(e)
    else:
        print('correct')


def master_solution(x: List[int]) -> List[int]:
    """Авторское решение"""
    return (
        pd.Series(x, dtype='int32')
            .mode()
            .tolist()
    )


if __name__ == '__main__':
    main()
