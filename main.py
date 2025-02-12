from __future__ import annotations
import time
import math
# from lib2to3.fixes.fix_input import context

import sympy

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
# 导入外部模块中的函数
def df(f, x):
    ''' Derivative of a given function f(x)

    Parameters
    ----------------
    f: function f(x) for which f'(x) is to be found.
    x: variable of differentiation

    Returns
    ----------------
    function f'(x) for the given f(x)
    '''
    return diff(f, x)


def newton(f, x: symbols, x0: int, epsilon, max_iter: int):
    '''Approximate solution of f(x)=0 by Newton Raphson Method.

    Parameters
    ----------------
    f : function for which we are trying to approximate the solution at f(x)=0
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations

    Returns
    ----------------
    integer
        Solution for f(x) = 0 as obtained by the Newton Raphson Method
    '''
    xn = x0
    ddx = df(f, x)
    for n in range(0, max_iter):
        fxn = f.subs(x, xn)
        if abs(fxn) < epsilon:
            print("Solution found after ", n, "iterations.")
            return xn
        dfxn = ddx.subs(x, xn)
        if dfxn == 0:
            print("Zero Derivative, no solution found.")
            return None
        xn = xn - fxn / dfxn
    print("Maximum Iterations exceeded. No solutions found.")
    return None

def edge_weight(mln: str):
    if mln.endswith('.mln'):
        with open(mln, 'r') as f:
            input_content = f.read()
        mln_problem = mln_parse(input_content)
    wfomcs_problem = MLN_to_WFOMC(mln_problem, '@F')
    weights = wfomcs_problem.weights
    preds = weights.keys()
    syms = create_vars('x0:{}'.format(len(preds)))  # 创建未知数
    context = WFOMCContext(wfomcs_problem)
    pred2weight = {}
    for sym, pred in zip(syms, preds):
        pred2weight[pred] = (sym, 1)  # False的weight赋1
    context.weights.update(pred2weight)
    Z = standard_wfomc(
        context.formula, context.domain, context.get_weight
    )
    count_dist = {}
    Z = expand(Z)
    for degrees, coef in coeff_dict(Z, syms):
        count_dist[degrees] = coef
    # f_wight代表未知数为weight下平均边的条数
    f_wight = 0.0
    for key in count_dist:
        counting = 1
        for i in range(int(key[0])):
            counting = counting * syms[0]
        f_wight = f_wight + 0.5 * key[0] * count_dist[key] * counting / Z
    return [f_wight, syms[0]]





# 从mlnproblem构造sentence，如何hard_rule为T，则构造硬约束的非；否则正常构造等价sentence
def mln_sentence(mln: MLNProblem, hard_rule: bool = True, pred_new: str = AUXILIARY_PRED_NAME):
    weightings: dict[Pred, tuple[Rational, Rational]] = dict()
    if hard_rule:
        sentence = bot
        for weighting, formula in zip(*mln.rules):
            if weighting != float('inf'):
                continue
            free_vars = formula.free_vars()
            for free_var in free_vars:
                formula = QuantifiedFormula(Universal(free_var), formula)
            sentence = sentence | (~formula)
    else:
        sentence = top
        for weighting, formula in zip(*mln.rules):
            free_vars = formula.free_vars()
            if weighting != float('inf'):
                aux_pred = new_predicate(len(free_vars), pred_new)
                formula = Equivalence(formula, aux_pred(*free_vars))
                # weightings[aux_pred] = (Rational(Fraction(math.exp(weighting)).numerator,
                #                                  Fraction(math.exp(weighting)).denominator), Rational(1, 1))
                # numerator, denominator = float.as_integer_ratio(weighting)
                # weightings[aux_pred] = (Rational(weighting, 1), Rational(1, 1))
                # weightings[aux_pred] = (Rational(Fraction(weighting).numerator,
                #                                  Fraction(weighting).denominator), Rational(1, 1))
                weightings[aux_pred] = (Rational(1, 1), Rational(1, 1))
            # 给free_var加上全称量词
            for free_var in free_vars:
                formula = QuantifiedFormula(Universal(free_var), formula)
            sentence = sentence & formula
    return [sentence, weightings]

def sentence_WFOMCSProblem(sentence1, weightings1, sentence2, weightings2, domain, cardinality_constraint = None):
    sentence = sentence1 & sentence2
    sentence = to_sc2(sentence)
    weightings = {**weightings1, **weightings2}
    return WFOMCSProblem(sentence, domain, weightings, cardinality_constraint)

def count_distribution_(context: WFOMCContext, preds1: list[Pred], preds2: list[Pred], mode: int,
                       algo: Algo = Algo.FASTERv2) \
        -> dict[tuple[int, ...], Rational]:
    #pred2weight = {}
    pred2sym = {}
    preds = preds1+preds2
    preds3 = []     #  指定算哪部分的wmc
    # if mode == 1:
    #     preds3 = preds1
    # else:
    #     preds3 = preds2
    syms = create_vars('x0:{}'.format(len(preds)))#创建未知数
    for sym, pred in zip(syms, preds):
        #if pred in pred2weight:
        #    continue
        pred2sym[pred] = sym
        #pred2weight[pred] = context.get_weight(pred)
        #pred2weight[pred][0] = pred2weight[pred][0] * sym
    # context.weights.update(pred2weight)

    def get_weight(pred):
        if pred not in pred2sym.keys():
            return context.get_weight(pred)
        return context.get_weight(pred)[0]*pred2sym[pred], context.get_weight(pred)[1]

    if algo == Algo.STANDARD:
        res = standard_wfomc(
            context.formula, context.domain, get_weight
        )
    elif algo == Algo.FASTERv2:
        res = faster_wfomc(
            context.formula, context.domain, get_weight, True
        )
    symbols = [pred2sym[pred] for pred in preds]
    count_dist = {}
    res = expand(res)
    # print(res)
    if context.decode_result(res) == 0:
        return {(0, 0): Rational(0, 1)}
    for degrees, coef in coeff_dict(res, symbols):
        count_dist[degrees] = coef
    return count_dist

# 输出俩mln的TVdistance和两个mln对应的属性（权重或边平均个数）
def MLN_TV(mln1: str,mln2: str, w1, w2) -> [float, float, float]:
    if mln1.endswith('.mln'):
        with open(mln1, 'r') as f:
            input_content = f.read()
        mln_problem1 = mln_parse(input_content)
    # 改变权重
    for i in range(len(mln_problem1.rules[1])):
        if mln_problem1.rules[0][i] == float('inf'):
            continue
        mln_problem1.rules[0][i] = w1

    wfomcs_problem11 = MLN_to_WFOMC1(mln_problem1, '@F')
    context11 = WFOMCContext(wfomcs_problem11)

    if mln2.endswith('.mln'):
        with open(mln2, 'r') as f:
            input_content = f.read()
        mln_problem2 = mln_parse(input_content)
    # 改变权重
    for i in range(len(mln_problem2.rules[1])):
        if mln_problem2.rules[0][i] == float('inf'):
            continue
        mln_problem2.rules[0][i] = w2

    wfomcs_problem22 = MLN_to_WFOMC1(mln_problem2, '@S')
    context22 = WFOMCContext(wfomcs_problem22)
    # Z1 = standard_wfomc(
    #     context11.formula, context11.domain, context11.get_weight
    # )
    # Z2 = standard_wfomc(
    #     context22.formula, context22.domain, context22.get_weight
    # )
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
def MLN_TV2(mln1: str,mln2: str):
    if mln1.endswith('.mln'):
        with open(mln1, 'r') as f:
            input_content = f.read()
        mln_problem1 = mln_parse(input_content)
    wfomcs_problem11 = MLN_to_WFOMC(mln_problem1)
    # context11 = WFOMCContext(wfomcs_problem11)

    if mln2.endswith('.mln'):
        with open(mln2, 'r') as f:
            input_content = f.read()
        mln_problem2 = mln_parse(input_content)
    wfomcs_problem22 = MLN_to_WFOMC(mln_problem2)
    # context22 = WFOMCContext(wfomcs_problem22)

    Z1 = wfomc(wfomcs_problem11, Algo.FASTERv2)
    Z2 = wfomc(wfomcs_problem22, Algo.FASTERv2)

    ma = max(Z1, Z2)
    mi = min(Z1, Z2)
    if mi != 0:
        TV = 0.5*(mi*(1/mi - 1/ma)+(ma-mi)/ma)
    elif mi == 0 and ma != 0:
        TV = 1
    else:
        TV = 0
    return float(TV)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # mln1 = "models\\E-R1.mln"
    # mln2 = "models\\E-R2.mln"
    # mln1 = "models\\deskmate.mln"
    # mln2 = "models\\deskmate.mln"
    mln1 = "models\\k_colored_graph_3.mln"
    mln2 = "models\\k_colored_graph_3.1.mln"

    # mln1 = "models\\employment.mln"
    # mln2 = "models\\employment.mln"
    # mln1 = "models\\exists-friends-smokes.mln"
    # mln2 = "models\\exists-friends-smokes.mln"
    # mln1 = "models\\friends-smokes.mln"
    # mln2 = "models\\friends-smokes.mln"
    # mln1 = "models\\weightedcolors.mln"
    # mln2 = "models\\weightedcolors.mln"
    vertex = 20
    m = math.floor(vertex/3)*math.ceil(vertex/3)+1
    # v1 = [0.5 + 0.5 * i for i in range(2 * m)]
    # v2 = np.linspace(math.ceil(vertex / 2), m, num=2 * m, endpoint=True)
    # weight1 = [0.1*(i+1) for i in range(20)]
    # weight2 = [0.1*(i+1) for i in range(20)]
    # [f_weight1, x1] = edge_weight(mln1)
    # [f_weight2, x2] = edge_weight(mln2)
    #
    # f_weight1 = sympy.simplify(f_weight1)
    # f_weight2 = sympy.simplify(f_weight2)
    #
    # for i in range(2 * m):
    #     w1[i] = newton(f_weight1 - v1[i], x1, 1, 0.001, 100)
    #     w2[i] = newton(f_weight2 - v2[i], x2, 1, 0.001, 100)

    # combinations = list(itertools.product(weight1, weight2))
    result = []
    # res.append(MLN_TV(mln1, mln2, float(w1[0]), float(w2[0])))

    # for w in combinations:
    #     res.append(MLN_TV(mln1, mln2, float(w[0]), float(w[1])))
    # 注意polinomial.py/coeff_dict函数里的处理
    # w1 = create_vars("w1")
    # w2 = create_vars("w2")
    last_w1 = '0'
    last_w2 = '0'
    for i in range(0, m+1):
        result.append([i, i, 0])
        with open(mln1, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        lines[9] = lines[9].replace(last_w1, str(i), 1)
        lines[10] = lines[10].replace(last_w1, str(i), 1)
        lines[11] = lines[11].replace(last_w1, str(i), 1)
        with open(mln1, 'w', encoding='utf-8') as file:
            file.writelines(lines)
        for j in range(i+1, m+1):
            with open(mln2, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            lines[9] = lines[9].replace(last_w2, str(j), 1)
            lines[10] = lines[10].replace(last_w2, str(j), 1)
            lines[11] = lines[11].replace(last_w2, str(j), 1)
            with open(mln2, 'w', encoding='utf-8') as file:
                file.writelines(lines)
            res = MLN_TV2(mln1, mln2)
            result.append([i, j, res])
            result.append([j, i, res])
            last_w1 = str(i)
            last_w2 = str(j)


    # start_time = time.time()
    # [x, y, res] = MLN_TV(mln1, mln2, w1, w2)
    # end_time = time.time()
    #
    # # 计算运行时间
    # execution_time = end_time - start_time
    # print(f"k_colored_graph_1代码运行时间: {execution_time:.6f} 秒")
    # # print(res)
    #
    # start_time = time.time()
    # for w in combinations:
    #     result.append([w[0], w[1], res.subs({w1: w[0], w2: w[1]})])
    # end_time = time.time()
    # execution_time = end_time - start_time
    # # 打印结果
    # print("代码运行时间: ", execution_time)

    # 创建DataFrame，并指定列名
    df = pd.DataFrame(result, columns=["m1", "m2", "TV"])
    excel_filename = "k_color3_domain20_fast.xlsx"
    df.to_excel(excel_filename, index=False)


    # fig = go.Figure(data=[go.Scatter3d(
    #     x=res[:, 0],
    #     y=res[:, 1],
    #     z=res[:, 2],
    #     mode='markers',
    #     marker=dict(
    #         size=10,  # 点的大小
    #         color=res[:, 2],  # 使用 z 轴的值作为颜色
    #         colorscale='Viridis',  # 颜色渐变方案
    #         colorbar=dict(title='Z轴值'),  # 添加颜色条
    #         showscale=True  # 显示颜色条
    #     )
    # )])
    # # 设置图形标题和坐标轴标签
    # fig.update_layout(title='edge-domain7', scene=dict(
    #                 xaxis_title='E-R1',
    #                 yaxis_title='E-R2',
    #                 zaxis_title='TV'))
    # fig.write_html('domain_77.html')
    # fig.show()