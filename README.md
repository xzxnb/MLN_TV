# MLN_TV
首先该MLN_TV的代码需要链接https://github.com/xzxnb/lifted_sampling_fo2.git里面的代码，因为MLN_TV里的函数功能是在lifted_sampling_fo2库基础上实现的
下面介绍一下MLN_TV的代码结构：
1.models文件里面定义了两个随机图，E-R1是简单随机图，E(X,Y)表示X,Y之间有边，E-R2是无孤立点的随机图，主要的函数都在main.py文件中，
下面介绍main.py文件中的函数：
1.edge_weight函数：输入随机图，输出在该随机图下平均边的个数与E(x,y)的权重之间的关系，即f_wight，其中权重syms[0]是自变量
2.mln_sentence函数：从mlnproblem构造sentence，如果hard_rule为T，则构造硬约束的非；否则正常构造等价sentence
3.sentence_WFOMCSProblem函数：将两个sentence合并构造WFOMCSProblem，weightings为相应sentence的权重
4.count_distribution_函数：如果mode=1，输出为计算论文算法3中WFOMC（ψ2, ∆, w1, ¯w1），mode=2，输出为计算论文算法3中WFOMC（ψ3, ∆, w2, ¯w2）
5.MLN_TV函数：输入为两个随机图和对应E(X,Y)的权重，输出论文算法3中MLN_TV的值

# MLN_TV
Firstly, the MLN_TV code needs to link to the code in https://github.com/xzxnb/lifted_sampling_fo2.git, as the functionality of MLN_TV is implemented based on the lifted_sampling_fo2 library.
Below is an introduction to the code structure of MLN_TV:
1. The models file defines two random graphs, E-R1 is a simple random graph, E(X,Y) represents an edge between X and Y, and E-R2 is a random graph without isolated points. The main functions are in the main.py file.
Below is an introduction to the functions in the main.py file:
1. edge_weight function: Input a random graph, output the relationship between the average number of edges and the weight of E(x,y) under that random graph, i.e., f_weight, where the weight syms[0] is the independent variable.
2. mln_sentence function: Construct a sentence from mlnproblem, if hard_rule is T, construct the negation of hard constraints; otherwise, construct equivalent sentences normally.
3. sentence_WFOMCSProblem function: Combine two sentences to construct WFOMCSProblem, with weightings being the corresponding weights of the sentences.
4. count_distribution_ function: If mode=1, output is the calculation of WFOMC (ψ2, ∆, w1, ¯w1) in the paper algorithm 3; if mode=2, output is the calculation of WFOMC (ψ3, ∆, w2, ¯w2) in the paper algorithm 3.
5. MLN_TV function: Input two random graphs and the corresponding weights of E(X,Y), output the value of MLN_TV in the paper algorithm 3.





