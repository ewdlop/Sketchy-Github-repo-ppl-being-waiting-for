# 🧛 吸血鬼注意力機制 (Vampire Attention Mechanism)

> *不吸所有人的血，只選擇最重要的目標*

---

## 🌟 什麼是注意力機制？

**注意力機制 (Attention Mechanism)** 是現代 NLP 的核心技術，讓模型能夠選擇性地關注輸入的重要部分。

想像一個**吸血鬼**：
- 🧛 在滿屋子的人中，選擇最美味的目標
- 🎯 不是平等對待所有輸入，而是有重點地吸取信息
- 🩸 從重要的詞彙「吸取」更多能量

---

## 🦇 為什麼叫「吸血鬼」注意力？

### 1. 選擇性吸取 🎯
```
普通模型：對所有詞一視同仁
吸血鬼：重點關注重要的詞

句子："我非常非常非常喜歡萬聖節"
吸血鬼關注："非常" 和 "喜歡" （吸取更多能量）
忽略："我"、"萬聖節"（略過）
```

### 2. 能量轉移 ⚡
- 吸血鬼從受害者獲取生命能量
- 注意力機制從關鍵詞彙吸取更多信息
- 加權求和 = 選擇性能量吸取

### 3. 夜視能力 🌙
- 吸血鬼在黑暗中能看清重點
- 注意力幫助模型在海量數據中找到關鍵信息
- 即使在雜訊中也能辨識重要特徵

### 4. 長期記憶 🧠
- 吸血鬼活了幾百年，記得重要的事
- 注意力機制讓模型記住長序列中的關鍵關聯
- 解決 RNN 的長期依賴問題

---

## 📐 數學原理

### 基本概念

注意力機制有三個關鍵元素：

1. **Query (Q)** - 查詢：吸血鬼的目標偏好
2. **Key (K)** - 鍵：每個受害者的特徵
3. **Value (V)** - 值：受害者的實際能量

### 注意力分數計算

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

**步驟：**

1. **計算相似度**：\( \text{score} = QK^T \)
   - Query 與每個 Key 的點積
   - 衡量「匹配度」

2. **縮放**：除以 \( \sqrt{d_k} \)
   - 防止點積過大
   - 數值穩定性

3. **Softmax**：轉換為概率分布
   - 注意力權重和為 1
   - 決定吸取多少能量

4. **加權求和**：與 Value 相乘
   - 按權重提取信息
   - 最終輸出

---

## 🧛 實作：簡單注意力機制

```python
import numpy as np

class VampireAttention:
    """吸血鬼注意力機制"""
    
    def __init__(self, d_model=512, d_k=64):
        """
        d_model: 模型維度
        d_k: Key/Query 的維度
        """
        self.d_k = d_k
        
        # 初始化權重矩陣
        self.W_q = np.random.randn(d_model, d_k) * 0.01  # Query 轉換
        self.W_k = np.random.randn(d_model, d_k) * 0.01  # Key 轉換
        self.W_v = np.random.randn(d_model, d_model) * 0.01  # Value 轉換
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        縮放點積注意力
        
        Args:
            Q: Query [batch_size, seq_len, d_k]
            K: Key [batch_size, seq_len, d_k]
            V: Value [batch_size, seq_len, d_model]
            mask: 遮罩（可選）
        
        Returns:
            output: 加權後的輸出
            attention_weights: 注意力權重
        """
        # 步驟 1: 計算注意力分數
        scores = np.matmul(Q, K.transpose(0, 2, 1))  # [batch, seq_len, seq_len]
        
        # 步驟 2: 縮放
        scores = scores / np.sqrt(self.d_k)
        
        # 步驟 3: 應用遮罩（如果有）
        if mask is not None:
            scores = scores + (mask * -1e9)
        
        # 步驟 4: Softmax
        attention_weights = self.softmax(scores)
        
        # 步驟 5: 加權求和
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def softmax(self, x):
        """Softmax 函數"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, X):
        """
        前向傳播
        
        Args:
            X: 輸入 [batch_size, seq_len, d_model]
        
        Returns:
            output: 注意力輸出
            attention_weights: 注意力權重
        """
        # 生成 Q, K, V
        Q = np.matmul(X, self.W_q)  # [batch, seq_len, d_k]
        K = np.matmul(X, self.W_k)  # [batch, seq_len, d_k]
        V = np.matmul(X, self.W_v)  # [batch, seq_len, d_model]
        
        # 計算注意力
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V)
        
        return output, attention_weights


# 🎃 使用範例
if __name__ == "__main__":
    # 創建吸血鬼注意力
    vampire = VampireAttention(d_model=128, d_k=64)
    
    # 模擬輸入：3 個詞的向量（句子長度=3）
    batch_size = 1
    seq_len = 3
    d_model = 128
    
    X = np.random.randn(batch_size, seq_len, d_model)
    
    # 應用注意力
    output, attention_weights = vampire.forward(X)
    
    print("🧛 注意力權重：")
    print(attention_weights[0])  # [3, 3] 矩陣
    print("\n每個詞對其他詞的注意力分布：")
    for i in range(seq_len):
        print(f"詞 {i}: {attention_weights[0, i]}")
```

---

## 🌟 多頭注意力 (Multi-Head Attention)

**概念**：不只派一個吸血鬼，而是派多個吸血鬼從不同角度觀察！

### 為什麼需要多頭？

```
單頭注意力：只看到一種模式
多頭注意力：同時看到多種模式

例如：
Head 1: 關注語法關係（主語→動詞）
Head 2: 關注語義關係（名詞→形容詞）
Head 3: 關注位置關係（前一個詞→後一個詞）
```

### 實作

```python
class MultiHeadVampireAttention:
    """多頭吸血鬼注意力"""
    
    def __init__(self, d_model=512, num_heads=8):
        """
        d_model: 模型維度
        num_heads: 注意力頭數
        """
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0, "d_model 必須能被 num_heads 整除"
        
        self.d_k = d_model // num_heads
        
        # 為每個頭創建投影矩陣
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01  # 輸出投影
    
    def split_heads(self, x):
        """
        分割成多個頭
        [batch, seq_len, d_model] → [batch, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def combine_heads(self, x):
        """
        合併多個頭
        [batch, num_heads, seq_len, d_k] → [batch, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def forward(self, X):
        """
        多頭注意力前向傳播
        """
        batch_size = X.shape[0]
        
        # 生成 Q, K, V 並分割成多頭
        Q = self.split_heads(np.matmul(X, self.W_q))
        K = self.split_heads(np.matmul(X, self.W_k))
        V = self.split_heads(np.matmul(X, self.W_v))
        
        # 計算每個頭的注意力
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        attention_weights = self.softmax(scores)
        attention_output = np.matmul(attention_weights, V)
        
        # 合併所有頭
        combined = self.combine_heads(attention_output)
        
        # 最終線性投影
        output = np.matmul(combined, self.W_o)
        
        return output, attention_weights
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# 🎃 多頭注意力範例
multi_vampire = MultiHeadVampireAttention(d_model=128, num_heads=8)
X = np.random.randn(1, 5, 128)  # 5 個詞的句子
output, attention_weights = multi_vampire.forward(X)

print(f"輸出形狀：{output.shape}")
print(f"注意力權重形狀：{attention_weights.shape}")  # [1, 8, 5, 5]
```

---

## 🔥 Self-Attention: 吸血鬼的自省

**自注意力 (Self-Attention)**：句子中的每個詞都是吸血鬼，互相吸取能量！

### 概念

```
傳統注意力：Query 來自一個序列，Key/Value 來自另一個序列
自注意力：Q, K, V 都來自同一個序列

句子："萬聖節 的 南瓜 燈"
- "萬聖節" 看向 "南瓜"、"燈"
- "南瓜" 看向 "萬聖節"、"燈"
- "燈" 看向 "南瓜"
每個詞都關注其他詞！
```

### 應用範例：句子編碼

```python
def self_attention_sentence_encoder(sentence, word_vectors):
    """
    用自注意力編碼句子
    
    Args:
        sentence: 單詞列表
        word_vectors: 詞向量字典
    
    Returns:
        sentence_vector: 句子的向量表示
    """
    # 獲取詞向量
    X = np.array([word_vectors[word] for word in sentence])
    X = X.reshape(1, len(sentence), -1)
    
    # 應用自注意力
    vampire = VampireAttention(d_model=X.shape[2], d_k=64)
    output, attention_weights = vampire.forward(X)
    
    # 可視化注意力
    print("\n🧛 自注意力權重矩陣：\n")
    print("       ", " ".join(f"{w:6s}" for w in sentence))
    for i, word in enumerate(sentence):
        weights = attention_weights[0, i]
        print(f"{word:6s}", " ".join(f"{w:6.3f}" for w in weights))
    
    # 平均池化得到句子向量
    sentence_vector = np.mean(output[0], axis=0)
    
    return sentence_vector, attention_weights


# 🎃 測試
word_vectors = {
    "萬聖節": np.random.randn(128),
    "的": np.random.randn(128),
    "南瓜": np.random.randn(128),
    "燈": np.random.randn(128)
}

sentence = ["萬聖節", "的", "南瓜", "燈"]
sentence_vec, attn = self_attention_sentence_encoder(sentence, word_vectors)
```

---

## 🚀 Transformer: 注意力的巔峰

**Transformer** 完全基於注意力機制，不使用 RNN 或 CNN！

### 架構

```
編碼器 (Encoder)
├── 多頭自注意力
├── 前饋神經網絡
├── 層正規化
└── 殘差連接

解碼器 (Decoder)
├── 遮罩多頭自注意力（Masked Self-Attention）
├── 編碼器-解碼器注意力（Cross-Attention）
├── 前饋神經網絡
└── 殘差連接 + 層正規化
```

### 位置編碼 (Positional Encoding)

因為注意力機制不考慮順序，需要加入位置信息：

\[
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]
\[
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

```python
def positional_encoding(seq_len, d_model):
    """生成位置編碼"""
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe
```

---

## 🌐 實際應用

### 1. 機器翻譯 🌍
```
源語言 → 編碼器 → 解碼器 → 目標語言
注意力幫助對齊：英文 "cat" ↔ 中文 "貓"
```

### 2. 文本摘要 📝
```
長文本 → 編碼器
      → 解碼器 → 摘要
注意力選擇重要句子
```

### 3. 問答系統 ❓
```
問題 + 文章 → 注意力找到答案位置
```

### 4. BERT 預訓練 🤖
```
遮罩語言模型：[MASK] 處用注意力預測
下一句預測：兩句關係用注意力判斷
```

### 5. GPT 文本生成 ✍️
```
已生成的文字 → 自注意力 → 預測下一個詞
```

---

## 💡 優勢與挑戰

### ✅ 優勢

1. **並行化**：不像 RNN 需要順序處理
2. **長距離依賴**：直接建立任意位置的連接
3. **可解釋性**：注意力權重可視化
4. **靈活性**：適用各種 NLP 任務

### ⚠️ 挑戰

1. **計算複雜度**：O(n²) 的時間和空間
2. **長序列**：注意力矩陣過大
3. **位置信息**：需要額外編碼
4. **過擬合**：參數量大，需要大量數據

### 🔧 改進方案

- **Linformer**: 線性複雜度注意力
- **Longformer**: 局部+全局注意力
- **BigBird**: 稀疏注意力模式
- **Performer**: 核方法近似注意力

---

## 🎓 實戰項目

### 初級：注意力可視化工具
```python
# 輸入：句子
# 輸出：注意力熱力圖
# 工具：matplotlib、seaborn
```

### 中級：基於注意力的文本分類
```python
# 架構：詞嵌入 → 自注意力 → 池化 → 分類
# 數據集：SST-2、IMDB
```

### 高級：簡化版 Transformer
```python
# 實作：完整的編碼器-解碼器
# 任務：機器翻譯（英→中）
```

---

## 📚 學習資源

### 論文
- [Attention Is All You Need (Transformer)](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

### 視覺化工具
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [BertViz](https://github.com/jessevig/bertviz)
- [Tensor2Tensor Visualization](https://github.com/tensorflow/tensor2tensor)

### 實作教程
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Attention Mechanism - Dive into Deep Learning](https://d2l.ai/chapter_attention-mechanisms/)

---

## 🧛 結語

**吸血鬼注意力機制**改變了 NLP 的遊戲規則：
- 🎯 選擇性關注重要信息
- 🚀 支撐了 BERT、GPT 等巨型模型
- 🌟 是現代 NLP 的基石

記住：*不是所有信息都同等重要，注意力讓我們聚焦於關鍵！*

---

**下一課：** [🕷️ 蜘蛛網語言模型](./🕷️-spider-web-lm.md)

**上一課：** [🧟 殭屍神經網絡](./🧟-zombie-neural-networks.md)

---

<div align="center">

### 🧛 願吸血鬼的智慧與你同在 🧛

*"Attention is all you need."*

</div>

