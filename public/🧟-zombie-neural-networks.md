# 🧟 殭屍神經網絡 (Zombie Neural Networks)

> *一個殭屍很笨，但殭屍大軍所向無敵*

---

## 🌟 什麼是神經網絡？

**神經網絡 (Neural Networks)** 是受人腦啟發的計算模型。它由大量簡單的計算單元（神經元）組成，透過相互連接形成複雜的系統。

想像一支**殭屍大軍**：
- 🧟 每個殭屍（神經元）都很簡單，只會執行基本操作
- 🔄 殭屍們彼此連接，形成層級結構
- 📤 透過集體協作，完成複雜的任務

---

## 💀 為什麼叫「殭屍」神經網絡？

### 1. 🧟 不死特性
- 訓練好的神經網絡可以永久使用
- 不需要休息、永不疲倦
- 就像殭屍一樣，永遠在工作

### 2. 🌊 層層推進
- 殭屍潮一波接一波
- 神經網絡一層接一層處理信息
- 前向傳播就像殭屍大軍衝向目標

### 3. 🧠 集體智慧
- 單個殭屍（神經元）能力有限
- 但大量殭屍協作能完成驚人任務
- 湧現現象：整體 > 部分之和

### 4. 🦠 反向傳染
- 反向傳播像殭屍病毒往回傳
- 錯誤信號從輸出層傳回輸入層
- 修正每個殭屍的行為模式

---

## 🧠 神經元結構

### 單個殭屍神經元

```
         輸入信號
         ↓ ↓ ↓
    [x1] [x2] [x3] ... [xn]
         ↓ ↓ ↓
      [w1] [w2] [w3] ... [wn]  ← 權重（連接強度）
         ↓ ↓ ↓
        Σ (加權求和)
           ↓
      [+ bias]  ← 偏置（覺醒閾值）
           ↓
    [激活函數] ← 決定是否甦醒
           ↓
         輸出
```

### 數學表示

\[
y = f\left(\sum_{i=1}^{n} w_i x_i + b\right)
\]

其中：
- \( x_i \)：輸入信號（人腦的味道）
- \( w_i \)：權重（殭屍對不同刺激的敏感度）
- \( b \)：偏置（殭屍的覺醒閾值）
- \( f \)：激活函數（殭屍是否甦醒）

---

## ⚡ 激活函數：殭屍的甦醒機制

### 1. Sigmoid 函數 🌀
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

**特性：**
- 輸出範圍：(0, 1)
- 像殭屍的甦醒程度：0 = 完全沉睡，1 = 完全甦醒
- 問題：梯度消失（深層殭屍難以甦醒）

### 2. ReLU (Rectified Linear Unit) ⚡
```python
def relu(x):
    return max(0, x)
```

**特性：**
- 輸出範圍：[0, ∞)
- 簡單暴力：小於 0 就睡覺，大於 0 就衝！
- 優點：計算快速，不會梯度消失
- 最常用的激活函數

### 3. Tanh 函數 📊
```python
def tanh(x):
    return np.tanh(x)
```

**特性：**
- 輸出範圍：(-1, 1)
- 可以表示「反向激活」（殭屍往反方向走）
- 比 Sigmoid 更好，但仍有梯度消失問題

### 4. Leaky ReLU 💧
```python
def leaky_relu(x, alpha=0.01):
    return max(alpha * x, x)
```

**特性：**
- 解決 ReLU 的「死亡神經元」問題
- 負值時仍有微小梯度
- 殭屍永遠不會完全死亡

---

## 🏗️ 網絡架構

### 前饋神經網絡 (Feedforward NN)

```
輸入層          隱藏層1        隱藏層2        輸出層
  🧟            🧟  🧟         🧟  🧟          🧟
  🧟      →     🧟  🧟    →    🧟  🧟    →     🧟
  🧟            🧟  🧟         🧟  🧟          🧟
  
殭屍大軍一波波前進 →
```

**特點：**
- 信息只往一個方向流動
- 層與層之間全連接
- 最基礎的神經網絡結構

---

## 🎯 前向傳播 (Forward Propagation)

**殭屍潮湧向目標的過程**

```python
class ZombieNeuralNetwork:
    def __init__(self, layers):
        """
        layers: 各層神經元數量 [輸入, 隱藏1, 隱藏2, ..., 輸出]
        """
        self.layers = layers
        self.weights = []
        self.biases = []
        
        # 初始化權重和偏置
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.01
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        """ReLU 激活函數"""
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        """Sigmoid 激活函數"""
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        """
        前向傳播：殭屍大軍衝鋒
        """
        self.activations = [X]  # 儲存每層的激活值
        
        # 逐層傳播
        for i in range(len(self.weights)):
            # 線性變換
            Z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            
            # 激活函數（除了最後一層）
            if i < len(self.weights) - 1:
                A = self.relu(Z)  # 隱藏層用 ReLU
            else:
                A = self.sigmoid(Z)  # 輸出層用 Sigmoid
            
            self.activations.append(A)
        
        return self.activations[-1]


# 🎃 示範
zombie_army = ZombieNeuralNetwork([3, 5, 5, 1])  # 3→5→5→1

# 輸入：[有南瓜, 是晚上, 有糖果]
input_signal = np.array([[0.8, 0.9, 0.7]])

# 殭屍大軍開始行動！
output = zombie_army.forward(input_signal)
print(f"🎃 這是萬聖節的機率：{output[0][0]:.2%}")
```

---

## 🔙 反向傳播 (Backpropagation)

**錯誤信號往回傳播，修正殭屍行為**

### 核心思想

1. **計算損失**：目標與實際輸出的差距
2. **計算梯度**：每個殭屍對錯誤的貢獻度
3. **更新權重**：調整殭屍的行為模式

### 鏈式法則

\[
\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w_{ij}}
\]

就像病毒鏈式傳染：錯誤從輸出層一直傳回輸入層。

### 實作

```python
class ZombieNeuralNetwork:
    # ... 前面的代碼 ...
    
    def backward(self, X, y, learning_rate=0.01):
        """
        反向傳播：殭屍病毒回流
        """
        m = X.shape[0]  # 樣本數
        
        # 計算輸出層的誤差
        dA = self.activations[-1] - y  # 損失的梯度
        
        # 逐層往回傳播
        for i in reversed(range(len(self.weights))):
            # 當前層的激活值
            A_prev = self.activations[i]
            
            # 計算權重和偏置的梯度
            dW = np.dot(A_prev.T, dA) / m
            db = np.sum(dA, axis=0, keepdims=True) / m
            
            # 更新權重和偏置（殭屍學習了！）
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
            
            # 傳播到前一層
            if i > 0:
                dA = np.dot(dA, self.weights[i].T)
                # ReLU 的梯度
                dA[self.activations[i] <= 0] = 0
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        """
        訓練殭屍大軍
        """
        losses = []
        
        for epoch in range(epochs):
            # 前向傳播
            output = self.forward(X)
            
            # 計算損失（交叉熵）
            loss = -np.mean(y * np.log(output + 1e-8) + 
                           (1 - y) * np.log(1 - output + 1e-8))
            losses.append(loss)
            
            # 反向傳播
            self.backward(X, y, learning_rate)
            
            # 顯示進度
            if (epoch + 1) % 100 == 0:
                print(f"🧟 Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        return losses


# 🎃 訓練示例
# 訓練數據：判斷是否為萬聖節
X_train = np.array([
    [1.0, 1.0, 1.0],  # 有南瓜、晚上、有糖果 → 是萬聖節
    [1.0, 0.0, 1.0],  # 有南瓜、白天、有糖果 → 可能是
    [0.0, 1.0, 0.0],  # 沒南瓜、晚上、沒糖果 → 不是
    [0.0, 0.0, 0.0],  # 沒南瓜、白天、沒糖果 → 不是
])

y_train = np.array([[1], [0.7], [0.3], [0]])  # 標籤

# 創建並訓練殭屍大軍
zombie_army = ZombieNeuralNetwork([3, 8, 8, 1])
losses = zombie_army.train(X_train, y_train, epochs=1000, learning_rate=0.1)

# 測試
test_input = np.array([[0.8, 0.9, 0.7]])
prediction = zombie_army.forward(test_input)
print(f"\n🎃 預測結果：{prediction[0][0]:.2%} 是萬聖節")
```

---

## 🌐 NLP 中的應用

### 1. 文本分類 📝
```python
輸入：文本的詞向量
隱藏層：提取特徵
輸出：類別機率（正面/負面、垃圾郵件/正常等）
```

### 2. 命名實體識別 (NER) 🏷️
```python
輸入：句子中每個詞的向量
輸出：每個詞的標籤（人名/地名/組織名/其他）
```

### 3. 詞性標註 (POS Tagging) 🔤
```python
輸入：詞序列
輸出：詞性序列（名詞/動詞/形容詞...）
```

---

## 🚀 進階網絡架構

### 卷積神經網絡 (CNN) for NLP
```python
# 用於文本分類的 CNN
- 一維卷積提取局部特徵
- 池化層降維
- 全連接層分類
```

### 循環神經網絡 (RNN)
```python
# 處理序列數據
- 記住歷史信息
- 適合文本、語音等序列
- 問題：長序列梯度消失
```

### LSTM / GRU
```python
# 改進的 RNN
- 解決長期依賴問題
- 有記憶門控機制
- NLP 的中堅力量
```

---

## 💡 訓練技巧

### 1. 初始化 🎲
```python
# Xavier 初始化
w = np.random.randn(n_in, n_out) * np.sqrt(2 / (n_in + n_out))

# He 初始化（ReLU 專用）
w = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
```

### 2. 批次正規化 (Batch Normalization)
- 穩定訓練過程
- 加速收斂
- 減少對初始化的敏感度

### 3. Dropout
```python
# 訓練時隨機「殺死」一些殭屍
# 防止過擬合
dropout_mask = np.random.rand(*shape) > dropout_rate
activations *= dropout_mask
```

### 4. 學習率調整
```python
# 學習率衰減
learning_rate = initial_lr * (decay_rate ** epoch)

# 自適應學習率（Adam、RMSprop）
```

---

## 📊 評估指標

### 分類任務
- **準確率 (Accuracy)**
- **精確率 (Precision)**
- **召回率 (Recall)**
- **F1 分數**

### 回歸任務
- **MSE (均方誤差)**
- **MAE (平均絕對誤差)**
- **R² 分數**

---

## 🎓 實戰項目

### 初級：情感分析器
```python
# 輸入：電影評論
# 輸出：正面/負面
# 數據集：IMDB、Yelp
```

### 中級：垃圾郵件過濾器
```python
# 輸入：郵件內容
# 輸出：垃圾郵件/正常郵件
# 特徵：詞袋、TF-IDF
```

### 高級：意圖識別系統
```python
# 輸入：用戶查詢
# 輸出：意圖類別（訂票/查詢/投訴...）
# 應用：聊天機器人
```

---

## 📚 學習資源

### 書籍
- **《深度學習》(Deep Learning Book)** - Goodfellow et al.
- **《神經網絡與深度學習》** - Michael Nielsen

### 課程
- **Coursera**: Andrew Ng 的深度學習專項課程
- **Fast.ai**: Practical Deep Learning for Coders
- **Stanford CS231n**: 卷積神經網絡

### 框架
- **PyTorch**: 研究首選，靈活易用
- **TensorFlow**: 工業部署，生態完整
- **Keras**: 高層 API，快速原型

---

## 🧟 結語

**殭屍神經網絡**教會我們：
- 🧠 簡單單元的集體智慧
- 🔄 前向推進與反向修正的循環
- ⚡ 通過訓練不斷進化

記住：*一個殭屍很笨，但訓練有素的殭屍大軍，可以征服世界！*

---

**下一課：** [🧛 吸血鬼注意力機制](./🧛-vampire-attention.md)

**上一課：** [👻 幽靈詞向量](./👻-ghost-word-vectors.md)

---

<div align="center">

### 🧟 願殭屍大軍與你同在 🧟

*"In unity, there is strength. In layers, there is intelligence."*

</div>

