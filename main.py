from __future__ import annotations

from lib2to3.fixes.fix_input import context

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
from sampling_fo2.problems import WFOMCSProblem, MLN_to_WFOMC
import copy

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
from fractions import Fraction
import plotly.graph_objects as go
from sympy import diff, symbols
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
    weightings: dict[Pred, tuple[Rational, Rational]] = dict()
    x = symbols('x')
    sentence = top
    for weighting, formula in zip(*mln_problem.rules):
        free_vars = formula.free_vars()
        if weighting != float('inf'):
            aux_pred = new_predicate(len(free_vars), '@F')
            formula = Equivalence(formula, aux_pred(*free_vars))
            weightings[aux_pred] = (Rational(Fraction(weighting).numerator,
                                             Fraction(weighting).denominator), Rational(1, 1))
        # 给free_var加上全称量词
        for free_var in free_vars:
            formula = QuantifiedFormula(Universal(free_var), formula)
        sentence = sentence & formula
    context = WFOMCSProblem(sentence, mln_problem.domain, weightings, None)
    pred2weight = {}






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
                weightings[aux_pred] = (Rational(Fraction(weighting).numerator,
                                                 Fraction(weighting).denominator), Rational(1, 1))
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
                       algo: Algo = Algo.STANDARD) \
        -> dict[tuple[int, ...], Rational]:
    context_c = copy.deepcopy(context)
    pred2weight = {}
    pred2sym = {}
    preds = list(set(preds1+preds2))
    preds3 = [] #指定算哪部分的wmc
    if mode==1:
        preds3 = preds1
    else:
        preds3 = preds2
    syms = create_vars('x0:{}'.format(len(preds)))#创建未知数
    for sym, pred in zip(syms, preds):
        if pred in pred2weight:
            continue
        weight = context_c.get_weight(pred)
        if pred in preds3:
            pred2weight[pred] = (weight[0] * sym, 1)#False的weight赋1
        else:
            pred2weight[pred] = (sym, 1)      #preds2的先不管
        pred2sym[pred] = sym

    context_c.weights.update(pred2weight)


    aa = context_c.weights

    if algo == Algo.STANDARD:
        res = standard_wfomc(
            context_c.formula, context_c.domain, context_c.get_weight
        )
    elif algo == Algo.FASTER:
        res = faster_wfomc(
            context_c.formula, context_c.domain, context_c.get_weight
        )
    symbols = [pred2sym[pred] for pred in preds]
    count_dist = {}
    res = expand(res)
    if context_c.decode_result(res) == 0:
        return {(0, 0): Rational(0, 1)}
    for degrees, coef in coeff_dict(res, symbols):
        count_dist[degrees] = coef
    return count_dist

# 输出俩mln的TVdistance和两个mln对应的属性（权重或边平均个数）
def MLN_TV(mln1: str,mln2: str, w1:float, w2:float) -> [float, float, float]:
    if mln1.endswith('.mln'):
        with open(mln1, 'r') as f:
            input_content = f.read()
        mln_problem1 = mln_parse(input_content)
    # 改变权重
    for i in range(len(mln_problem1.rules[1])):
        if mln_problem1.rules[0][i] == float('inf'):
            continue
        mln_problem1.rules[0][i] = w1

    wfomcs_problem11 = MLN_to_WFOMC(mln_problem1, '@F')
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

    wfomcs_problem22 = MLN_to_WFOMC(mln_problem2, '@S')
    context22 = WFOMCContext(wfomcs_problem22)
    Z1 = standard_wfomc(
        context11.formula, context11.domain, context11.get_weight
    )
    Z2 = standard_wfomc(
        context22.formula, context22.domain, context22.get_weight
    )

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

    # 分别为包含俩硬约束和只包含其中一个硬约束的情况
    context1 = WFOMCContext(wfomcs_problem1)
    context2 = WFOMCContext(wfomcs_problem2)
    context3 = WFOMCContext(wfomcs_problem3)

    count_dist1 = count_distribution_(context1, list(weights1.keys()), list(weights2.keys()), 1)
    count_dist2 = count_distribution_(context1, list(weights1.keys()), list(weights2.keys()), 2)

    res = Rational(0, 1)
    # x, y分别代表两个mln在各自weight下平均边的条数
    x = 0.0
    y = 0.0
    # 同时满足第一个mln和第二个mln硬约束的情况
    for key in count_dist1:
        x = x + key[0]*count_dist1[key] / Z1
        y = y + key[1]*count_dist2[key] / Z2
        res = res + abs(count_dist1[key] / Z1 - count_dist2[key] / Z2)

    # 不满足第一个mln的硬约束加上第二个mln
    count_dist3 = count_distribution_(context2, list(weights1_hard.keys()), list(weights2.keys()), 2)
    for key in count_dist3:
        y = y + key[1]*count_dist3[key] / Z2
        res = res + abs(count_dist3[key] / Z2)

    # 不满足第二个mln的硬约束加上第一个mln
    count_dist4 = count_distribution_(context3, list(weights1.keys()), list(weights2_hard.keys()), 1)
    for key in count_dist4:
        x = x + key[0] * count_dist4[key] / Z1
        res = res + abs(count_dist4[key] / Z1)
    x = float(round_rational(x))/2
    y = float(round_rational(y))/2
    res = 0.5*float(round_rational(res))
    # return [w1, w2, res]
    return [x, y, res]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mln1 = "models\\E-R1.mln"
    mln2 = "models\\E-R2.mln"
    vertex:int  = 3
    m = int(vertex*(vertex-1)/2)
    v1 = [0.5 + 0.5 * i for i in range(2 * m)]
    v2 = [0.5 + 0.5 * i for i in range(2 * m)]



    w1 = [0.2 + i*0.2 for i in range(20)]
    w2 = [0.2 + i*0.2 for i in range(20)]
    combinations = list(itertools.product(w1, w2))
    res = []
    for w in combinations:
        res.append(MLN_TV(mln1, mln2, w[0], w[1]))
    for a in res:
        print(res)
    res = np.array(res)
    # 创建三维图
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # 绘制散点图
    # ax.scatter(res[:, 0], res[:, 1], res[:, 2], s=100, c=res, cmap='viridis')
    # # 设置标签
    # ax.set_xlabel('E-R1')
    # ax.set_ylabel('E-R2')
    # ax.set_zlabel('TV')
    # ax.set_title('0-0.2-4')
    # fig.write_html("3d_scatter.html")
    # plt.savefig('TV0_4.png')
    # # 显示图形
    # plt.show()
    # fig = go.Figure(data=[go.Scatter3d(x=res[:, 0], y=res[:, 1], z=res[:, 2], mode='markers')])
    fig = go.Figure(data=[go.Scatter3d(
        x=res[:, 0],
        y=res[:, 1],
        z=res[:, 2],
        mode='markers',
        marker=dict(
            size=10,  # 点的大小
            color=res[:, 2],  # 使用 z 轴的值作为颜色
            colorscale='Viridis',  # 颜色渐变方案
            colorbar=dict(title='Z轴值'),  # 添加颜色条
            showscale=True  # 显示颜色条
        )
    )])
    # 设置图形标题和坐标轴标签
    fig.update_layout(title='edge-domain7', scene=dict(
                    xaxis_title='E-R1',
                    yaxis_title='E-R2',
                    zaxis_title='TV'))
    fig.write_html('domain_7.html')
    fig.show()