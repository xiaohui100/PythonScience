#!/usr/bin/env python  
# encoding: utf-8  
"""
@author: Alfons
@contact: alfons_xh@163.com
@file: NumpyExample.py 
@time: 2018/1/18 20:05 
@version: v1.0 
"""
import numpy as np


print("-" * 40 + "Numpy普通数组" + "-" * 40)
a = np.array(np.random.randint(0, 10, 5))
print("np.array(np.random.randint(0, 10, 5)) = %s" % a)
print("a.max() = %s" % a.max())
print("a.dtype = %s" % a.dtype)     # 元素类型
print("a.mean() = %s" % a.mean())     # 元素平均值
print("np.median(a) = %s" % np.median(a))     # 元素的中位数
print("a.std() = %s" % a.std())     # 标准差
print("a.sum() = %s" % a.sum())     # 各元素和
print("a.cumsum() = %s" % np.cumsum(a))     # 各元素累加，[a, b, c].cumsum() = [a, a + b, a + b + c]


a = np.array([
    np.arange(1, 10, 2),
    np.linspace(0, 1, 5),   # 起点、终点、数据点
    np.linspace(0, 1, 5, endpoint = False),  # 起点、终点、数据点，终点不算
    np.linspace(0, 1, 5, endpoint = True)  # 起点、终点、数据点，终点算
])
print("""np.array([
    np.arange(1, 10, 2),
    np.linspace(0, 1, 5),   # 起点、终点、数据点
    np.linspace(0, 1, 5, endpoint = False),  # 起点、终点、数据点，终点不算
    np.linspace(0, 1, 5, endpoint = True)  # 起点、终点、数据点，终点算
]) = \n%s""" % a)
print("a.shape = {shape}".format(shape = a.shape))  # 数组形状
print("a.max = %s" % a.max())   # 数组最大值
print("a.ndim = %s" % a.ndim)   # 数组纬度
print("a.dtype = %s" % a.dtype)     # 元素类型

print("-" * 40 + "Numpy特殊数组" + "-" * 40)
a = np.ones(shape = (3, 3))  # 全1数组
print("np.ones(shape = (3, 3)) = \n%s" % a)
a = np.zeros(shape = (3, 3))  # 全0数组
print("np.zeros(shape = (3, 3)) = \n%s" % a)
a = np.eye(3)  # 对角矩阵
print("np.eye(3) = \n%s" % a)
a = np.eye(3, k = 1)  # 对角矩阵
print("np.eye(3, k = 1) = \n%s" % a)


print("-" * 40 + "Numpy对角矩阵" + "-" * 40)
a = np.diag(np.array(range(1, 6)))  # 对角矩阵
print("np.diag(np.array(range(1, 6))) = \n%s" % a)

x = np.arange(0, 9).reshape((3, 3))
print("np.arange(0, 9).reshape((3, 3)) = \n%s" % x)
a = np.diag(x)
print("np.diag(x) = %s" % a)
a = np.diag(x, k = 1)
print("np.diag(x, k = 1) = %s" % a)
a = np.diag(x, k = 2)
print("np.diag(x, k = 2) = %s" % a)


print("-" * 40 + "Numpy随机数" + "-" * 40)
a = np.random.rand(4)    # [0, 1] 的均匀分布随机数
print("np.random.rand(4) = \n%s" % a)
a = np.random.rand(3, 3)  # 参数为shape
print("np.random.rand(3, 3) = \n%s" % a)


print("-" * 40 + "Numpy切片" + "-" * 40)
a = np.array([i + j * 10 for j in range(0, 6) for i in range(0, 6)]).reshape(6, 6)
print("np.array([i + j * 10 for j in range(0, 6) for i in range(0, 6)]).reshape(6, 6) = \n%s" % a)
print("a[3, 4] = %s" % a[3, 4])
print("a[3, :4] = %s" % a[3, :4])
print("a[4:, 4:] = \n%s" % a[4:, 4:])
print("a[:, 4] = %s" % a[:, 4])

a = np.arange(6) + np.arange(0, 51, 10)[:, np.newaxis]
print("np.arange(6) + np.arange(0, 51, 10)[:, np.newaxis] = \n%s" % a)

a = np.ones((4, 4), dtype = int)
a[2, 3] = 2
a[3, 1] = 6
print("np.ones((4, 4), dtype = int) = \n%s" % a)

a = np.delete(np.diag(np.arange(2, 7, dtype = float), k = -1), 5, 1)
print("np.delete(np.diag(np.arange(2, 7, dtype = float), k = -1), 5, 1) = \n%s" % a)

a = np.tile(np.array([[4, 3], [2, 1]]), (2, 3))
print("np.tile(np.array([[4, 3], [2, 1]]), (2, 3)) = \n%s" % a)


print("-" * 40 + "100以内素数筛选：埃拉托斯特尼筛法" + "-" * 40)
is_prime = np.ones((100,), dtype=bool)
is_prime[:2] = 0
N_max = int(np.sqrt(len(is_prime)))
for j in range(2, N_max):
    is_prime[2*j::j] = False
prime = [x for x in range(len(is_prime)) if is_prime[x]]
print("prime = \n%s" % prime)


print("-" * 40 + "Numpy象征索引" + "-" * 40)
np.random.seed(3)
a = np.random.randint(0, 20, 15)
print("np.random.randint(0, 20, 15) = %s" % a)
a = a[a % 3 == 0]
print("a = %s" % a)

a = np.arange(0, 11, 1)
print("a = %s" % a)
a = a[[2, 1, 2, 1, 2, 1]]
print("a[[2, 1, 2, 1, 2, 1]] = %s" % a)

a = np.arange(0, 11, 1)
a = a[np.array([[3, 4], [9, 7]])]
print("a[np.array([[3, 4], [9, 7]])] = \n%s" % a)

a = np.arange(6) + np.arange(0, 51, 10)[:, np.newaxis]
print("a = \n%s" % a)
b = a[(0, 1, 2, 3, 4), (1, 2, 3, 4, 5)]
print("a[(0, 1, 2, 3, 4), (1, 2, 3, 4, 5)] = %s" % b)


print("-" * 40 + "Numpy数组的数值操作" + "-" * 40)
a = np.arange(1, 5)
print("a = %s" % a)
print("a + 1 = %s" % (a + 1))
print("2 ** a = %s" % (2 ** a))

b = np.ones(4) + 1
print("a = %s" % a)
print("b = %s" % b)
print("a + b = %s " % (a + b))
print("a * b = %s " % (a * b))

c = np.ones((3, 3))
print("c = %s" % c)
print("c * c = \t\t# 数组相乘\n%s" % (c * c))
print("c.dot(c) = \t\t# 矩阵相乘\n%s" % (c.dot(c)))


print("-" * 40 + "Numpy矩阵变换" + "-" * 40)
a = np.triu(np.ones((4, 4)), k = 1)
print("np.triu(np.ones((4, 4)), k = 1) = \t\t# 上三角矩阵\n%s" % a)
print("a.T = \t\t# 矩阵的行列变换\n%s" % a.T)
b = np.tril(np.ones((4, 4)), k = -1)
print("np.tril(np.ones((4, 4)), k = -1) = \t\t# 下三角矩阵\n%s" % b)



print("-" * 40 + "Numpy添加纬度" + "-" * 40)
a = np.arange(4)
print("a = %s" % a)
print("a = %s" % a[:, np.newaxis])
print("a = %s" % a[np.newaxis, :])
pass