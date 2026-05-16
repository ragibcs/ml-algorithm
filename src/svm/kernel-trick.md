# Kernel Trick

Kernel হলো এমন একটি Technique যা Data-কে Higher Dimension-এ নিয়ে যায় যাতে Data সহজে আলাদা করা যায়।

## জনপ্রিয় Kernel Function

### 1. Linear Kernel

```math
K(x_i,x_j)=x_i^Tx_j
```

### 2. Polynomial Kernel

```math
K(x_i,x_j)=(x_i^Tx_j+c)^d
```

### 3. RBF Kernel

```math
K(x_i,x_j)=exp(-\gamma ||x_i-x_j||^2)
```

### 4. Sigmoid Kernel

```math
K(x_i,x_j)=tanh(\alpha x_i^Tx_j+c)
```
