from __future__ import annotations
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
# 导入外部模块中的函数
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
    if context_c.decode_result(res) ==0:
        return {'0':0}
    for degrees, coef in coeff_dict(res, symbols):
        count_dist[degrees] = coef
    return count_dist
def MLN_TV(mln1: str,mln2: str, w1:float, w2:float) -> Rational:
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


    context1 = WFOMCContext(wfomcs_problem1)
    context2 = WFOMCContext(wfomcs_problem2)
    context3 = WFOMCContext(wfomcs_problem3)

    count_dist1 = count_distribution_(context1, list(weights1.keys()), list(weights2.keys()), 1)
    count_dist2 = count_distribution_(context1, list(weights1.keys()), list(weights2.keys()), 2)

    res = Rational(0, 1)
    for key in count_dist1:
        res = res + abs(count_dist1[key] / Z1 - count_dist2[key] / Z2)

    count_dist3 = count_distribution_(context2, list(weights1_hard.keys()), list(weights2.keys()), 2)
    for key in count_dist3:
        res = res + abs(count_dist3[key] / Z2)

    count_dist4 = count_distribution_(context3, list(weights1.keys()), list(weights2_hard.keys()), 1)
    for key in count_dist4:
        res = res + abs(count_dist4[key] / Z1)
    return res

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mln1 = "models\\E-R1.mln"
    mln2 = "models\\E-R2.mln"
    w1 = [0.2 + i*0.2 for i in range(20)]
    w2 = [0.2 + i*0.2 for i in range(20)]

    combinations = list(itertools.product(w1, w2))
    res = []
    for w in combinations:
        res.append(round_rational(MLN_TV(mln1, mln2, w[0], w[1])))
    for a in res:
        print(res)

    x = [comb[0] for comb in combinations]
    y = [comb[1] for comb in combinations]

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    res = 0.5*np.array(res, dtype=float)
    i = 0
    for w in combinations:
        print('w[0]: ', w[0], 'w[1]: ', w[1], 'res', res[i])
        i = i+1
    print('i: ',i)
    # 创建三维图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    ax.scatter(x, y, res, s=100, c=res, cmap='viridis')

    # 设置标签
    ax.set_xlabel('E-R1')
    ax.set_ylabel('E-R2')
    ax.set_zlabel('TV')
    ax.set_title('0-0.2-4')
    plt.savefig('TV0_4.png')
    # 显示图形
    plt.show()


