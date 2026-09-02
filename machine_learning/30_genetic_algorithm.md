# 遗传算法 (Genetic Algorithm, GA)

## 1. 算法原理

遗传算法（GA）是一种模拟自然界生物进化机制（自然选择与遗传学）的全局优化算法。

核心步骤包含：

1. **编码/初始化**：将问题的解编码为染色体二进制位串。
2. **适应度评估**：评估个体适应度值。
3. **选择 (Selection)**：轮盘赌选择淘汰劣等个体。
4. **交叉 (Crossover)**：交换父代染色体片段生成子代。
5. **变异 (Mutation)**：低概率随机翻转位，维持多样性。

---

## 2. 数学公式与流程

1. **轮盘赌选择概率**：

$$P_i = \frac{f_i}{\sum_{j=1}^{N} f_j}$$

* $P_i$: 第 $i$ 个个体被选中遗传到下一代的概率
* $f_i$: 第 $i$ 个个体的适应度值
* $f_j$: 第 $j$ 个个体的适应度值
* $N$: 种群中的个体总数量

2. **单点交叉与按位变异**。

---

## 3. ASCII 流程图

```
     [ 初始化随机种群 ]
             |
             v
     [ 计算个体适应度 f_i ]
             |
             v
      [ 满足终止条件? ] ----(是)----> [ 输出最佳个体 ]
             |
            (否)
             v
     [ 选择 -> 交叉 -> 变异 ]
             |
             +-----------------------+


```

---

## 4. Python 代码实现 (纯 Python / NumPy)

```python
import numpy as np

POP_SIZE = 50
GEN_SIZE = 100
CHROMO_LEN = 20
CROSS_RATE = 0.8
MUTATE_RATE = 0.01

def decode_chromosome(pop):
    precision = 2**CHROMO_LEN - 1
    decimal = pop.dot(2 ** np.arange(CHROMO_LEN)[::-1])
    return decimal / precision

def fitness_func(x):
    return x * np.sin(10 * np.pi * x) + 2.0

def select(pop, fitness):
    probs = fitness / np.sum(fitness)
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=probs)
    return pop[idx]

def crossover(pop):
    new_pop = []
    for i in range(0, POP_SIZE, 2):
        p1, p2 = pop[i].copy(), pop[i+1].copy()
        if np.random.rand() < CROSS_RATE:
            point = np.random.randint(1, CHROMO_LEN)
            p1[point:], p2[point:] = p2[point:].copy(), p1[point:].copy()
        new_pop.append(p1)
        new_pop.append(p2)
    return np.array(new_pop)

def mutate(pop):
    for i in range(POP_SIZE):
        for j in range(CHROMO_LEN):
            if np.random.rand() < MUTATE_RATE:
                pop[i, j] = 1 - pop[i, j]
    return pop

if __name__ == "__main__":
    np.random.seed(42)
    pop = np.random.randint(0, 2, size=(POP_SIZE, CHROMO_LEN))
    
    for gen in range(GEN_SIZE):
        x = decode_chromosome(pop)
        fitness = fitness_func(x)
        
        pop = select(pop, fitness)
        pop = crossover(pop)
        pop = mutate(pop)

    best_x = decode_chromosome(pop)[np.argmax(fitness_func(decode_chromosome(pop)))]
    print(f"遗传算法搜索到的最优 x: {best_x:.4f}")
    print(f"最大目标函数值: {fitness_func(best_x):.4f}")

```