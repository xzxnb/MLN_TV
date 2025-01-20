import time
import math
from sampling_fo2.wfomc import standard_wfomc, faster_wfomc, Algo, wfomc

from sampling_fo2.problems import MLNProblem, WFOMCSProblem

from sampling_fo2.fol.sc2 import SC2, to_sc2
from sampling_fo2.fol.syntax import AtomicFormula, Const, Pred, top, AUXILIARY_PRED_NAME, \
    Formula, QuantifiedFormula, Universal, Equivalence, bot
from sampling_fo2.fol.utils import new_predicate
from sampling_fo2.utils.polynomial import coeff_dict, create_vars, expand

from sampling_fo2.utils import MultinomialCoefficients, multinomial, \
    multinomial_less_than, RingElement, Rational, round_rational

from sampling_fo2.context import WFOMCContext

from sampling_fo2.fol.syntax import Const, Pred, QFFormula, PREDS_FOR_EXISTENTIAL

from sampling_fo2.parser.mln_parser import parse as mln_parse
from sampling_fo2.problems import WFOMCSProblem, MLN_to_WFOMC, MLN_to_WFOMC1
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
from fractions import Fraction
import plotly.graph_objects as go
from sympy import diff, symbols
# from scipy.optimize import fsolve
from main import mln_sentence, sentence_WFOMCSProblem, count_distribution_
def MLN_TV(mln1: str,mln2: str, w1) -> [float, float, float]:
    if mln1.endswith('.mln'):
        with open(mln1, 'r') as f:
            input_content = f.read()
        mln_problem1 = mln_parse(input_content)
    # 改变权重
    for i in range(len(mln_problem1.rules[1])):
        if mln_problem1.rules[0][i] == float('inf'):
            continue
        if mln_problem1.rules[0][i] == 0:
            mln_problem1.rules[0][i] = w1

    wfomcs_problem11 = MLN_to_WFOMC1(mln_problem1, '@F')
    context11 = WFOMCContext(wfomcs_problem11)

    if mln2.endswith('.mln'):
        with open(mln2, 'r') as f:
            input_content = f.read()
        mln_problem2 = mln_parse(input_content)

    wfomcs_problem22 = MLN_to_WFOMC1(mln_problem2, '@S')
    context22 = WFOMCContext(wfomcs_problem22)

    Z1 = wfomc(context11, Algo.FASTERv2)
    Z2 = wfomc(context22, Algo.FASTERv2)

    weights1: dict[Pred, tuple[Rational, Rational]]
    weights1_hard: dict[Pred, tuple[Rational, Rational]]
    weights2: dict[Pred, tuple[Rational, Rational]]
    weights2_hard: dict[Pred, tuple[Rational, Rational]]

    domain = mln_problem1.domain
    [sentence1, weights1] = mln_sentence(mln_problem1, False, 'F')
    [sentence1_hard, weights1_hard] = mln_sentence(mln_problem1, True, 'F')
    [sentence2, weights2] = mln_sentence(mln_problem2, False, 'S')
    [sentence2_hard, weights2_hard] = mln_sentence(mln_problem2, True, 'S')

    wfomcs_problem1 = sentence_WFOMCSProblem(sentence1, weights1, sentence2, weights2, domain)
    wfomcs_problem2 = sentence_WFOMCSProblem(sentence1_hard, weights1_hard, sentence2, weights2, domain)
    wfomcs_problem3 = sentence_WFOMCSProblem(sentence1, weights1, sentence2_hard, weights2_hard, domain)
    print('wfomcs_problem1: ', wfomcs_problem1)

    # 分别为包含俩硬约束和只包含其中一个硬约束的情况
    context1 = WFOMCContext(wfomcs_problem1)
    context2 = WFOMCContext(wfomcs_problem2)
    context3 = WFOMCContext(wfomcs_problem3)

    count_dist1 = count_distribution_(context1, list(weights1.keys()), list(weights2.keys()), 1)
    # count_dist2 = count_distribution_(context1, list(weights1.keys()), list(weights2.keys()), 2)
    print('count_dist1: ', count_dist1)
    res = Rational(0, 1)
    # x, y分别代表两个mln在各自weight下平均边的条数
    x = 0.0
    y = 0.0
    # 同时满足第一个mln和第二个mln硬约束的情况
    for key in count_dist1:
        w = w1**key[0]/Z1 - w2**key[1]/Z2
        # x = key[0]*count_dist1[key] / Z1
        # y = y + key[1]*count_dist2[key] / Z2
        res = res + abs(w * count_dist1[key])

    # # 不满足第一个mln的硬约束加上第二个mln
    # count_dist3 = count_distribution_(context2, list(weights1_hard.keys()), list(weights2.keys()), 2)
    # for key in count_dist3:
    #     y = y + w2**key[1]*count_dist3[key] / Z2
    #     res = res + abs(count_dist3[key] / Z2)
    #
    # # 不满足第二个mln的硬约束加上第一个mln
    # count_dist4 = count_distribution_(context3, list(weights1.keys()), list(weights2_hard.keys()), 1)
    # for key in count_dist4:
    #     x = x + w1**key[0] * count_dist4[key] / Z1
    #     res = res + abs(count_dist4[key] / Z1)
    res = 0.5*res
    # x = float(round_rational(x))/2
    # y = float(round_rational(y))/2
    # res = 0.5 * float(round_rational(res))
    # return [w1, w2, res]
    return [x, y, res]
if __name__ == '__main__':
    mln1 = "models\\fr-sm-dr1.mln"
    mln2 = "models\\fr-sm-dr2.mln"
    weight1 = [0.1*(i+1) for i in range(20)]
    w1 = create_vars("w1")
    start_time = time.time()
    [x, y, res] = MLN_TV(mln1, mln2, w1, w2)
    end_time = time.time()

    # 计算运行时间
    execution_time = end_time - start_time
    print(f"k_colored_graph_1代码运行时间: {execution_time:.6f} 秒")
    # print(res)

    start_time = time.time()
    for w in combinations:
        result.append([w[0], w[1], res.subs({w1: w[0], w2: w[1]})])
    end_time = time.time()
    execution_time = end_time - start_time
    # 打印结果
    print("代码运行时间: ", execution_time)






