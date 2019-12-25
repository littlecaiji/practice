# Exercises

## Sigmoid neurons simulating perceptrons, part I
Question: Suppose we take all the weights and biases in a network of
perceptrons, and multiply them by a positive constant, c > 0.
Show that the behaviour of the network doesn't change.  

由于只是perceptrons Network，所以只需要证明一个perception不变，即可证明整个network不变。  
单个perception的表达式：
$$ output=\left\{
\begin{aligned}
1 & , \qquad if \quad w*x + b > 0 \\
0 & , \qquad if \quad w*x + b \le 0
\end{aligned}
\right.
$$

乘上任意一个大于0的常数c，只会变为
$$ output=\left\{
\begin{aligned}
1 & , \qquad if \quad c*w*x + c*b > 0 \\
0 & , \qquad if \quad c*w*x + c*b \le 0
\end{aligned}
\right.
$$
在不等式判断的两侧同除以c,就和原表达式相同了。

## Sigmoid neurons simulating perceptrons, part II
Question : 和part I相同的网络，但用sigmoid neurons替代perceptrons，一个前提条件：$\omega \cdot x\ not= 0$，证明当$c \rightarrow \infty$时，用c乘以所有的权重$\omega$和偏移量$b$，该sigmoid网络的表现和percptron网络相同。并解释$\omega \cdot x = 0$时为何不等价。

单个sigmoid单元的表达式如下：
$$
output = \frac{1}{1+exp(\omega \cdot x + b)} = \frac{1}{1+exp(\sum \omega_i * x_i + b_i)}
$$
在乘以一个正常数c的时候，表达式将变为

$$
output = \frac{1}{1+exp(c\times \sum \omega_i * x_i + c\times b_i)} = \frac{1}{1+exp[c\times(\sum \omega_i * x_i +  b_i)]}
$$
当$c \rightarrow \infty$时，$c\times(\sum \omega_i * x_i +  b_i)$为$+\infty$或者$-\infty$，进而对应着$exp(\omega \cdot x + b)$为$+\infty$还是$0$，进而对应着$output$为$0$还是$1$。而究竟是+还是-，取决于$(\sum \omega_i * x_i +  b_i)$的正负，这和perception单元是相同的表达式。

