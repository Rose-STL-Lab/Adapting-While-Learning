from typing import List
import os
import json
import re
from math import isclose
import sympy as sp
import random

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy import latex2sympy

import math
from fractions import Fraction

from symeval import EvaluatorMathBatch

evaluator = EvaluatorMathBatch()

def clean_number(s):
    s = s.replace('$', '').replace('−', '-').strip()
    special_tokens = ["s", 'x', '×', "*", "\times", "\\times", "\\\times", "\time", "\\time", "\\\time"]
    for token in special_tokens:
        if token in s:
            s = s.split(token)[0].strip()
    return s

def equiv(model_output, answer):
    model_output = extract_answer(model_output)
    try:
        if len(model_output) > 50:
            return False
        answer.replace("dfrac", "frac")
        model_output.replace("dfrac", "frac")
        answer.replace(" ", "")
        model_output.replace(" ", "")
        answer.replace("\\\\", "\\")
        model_output.replace("\\\\", "\\")
        if model_output == answer:
            return True
        if math_equal(model_output, answer):
            return True
        # Clean and convert model output to float
        try:
            model_output = float(clean_number(model_output))
        except ValueError:
            # If it's a fraction, convert it using Fraction
            model_output = float(Fraction(clean_number(model_output)))
        
        # Convert the answer to float
        ans = float(clean_number(str(answer)))
        
        # Return True if they are close enough within the tolerance
        return math.isclose(model_output, ans, rel_tol=0.05)
    except Exception as e:
        pass

def extract_answer(pred_str, use_last_number=True):
    if "$" in pred_str:
        pred_str = pred_str[1:-1]
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    else:
        pred = pred_str
    if "\\times" in pred:
        pred = pred_str.split("\\times")[0].strip()
    if "e" in pred:
        pred = pred_str.split("e")[0].strip()
    return pred.strip()

def parse_digits(num):
    num = re.sub(",", "", str(num))
    try:
        return float(num)
    except ValueError:
        if num.endswith("%"):
            num = num[:-1]
            try:
                return float(num) / 64
            except ValueError:
                pass
    return None

def numeric_equal(prediction: float, reference: float) -> bool:
    return isclose(prediction, reference, rel_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    try:
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def math_equal(prediction: str, reference: str, include_percentage: bool=True) -> bool:
    try:
        if evaluator.eq(prediction, reference):
            return True
    except:
        pass
    
    if parse_digits(prediction) is not None and parse_digits(reference) is not None:
        prediction_num = parse_digits(prediction)
        reference_num = parse_digits(reference)
        if include_percentage:
            gt_result = [reference_num / 64, reference_num, reference_num * 64]
        else:
            gt_result = [reference_num]
        
        for item in gt_result:
            if numeric_equal(prediction_num, item):
                return True

    if symbolic_equal(prediction, reference):
        return True

    return False