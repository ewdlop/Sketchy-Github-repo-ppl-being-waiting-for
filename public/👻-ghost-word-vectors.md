# 👻 幽靈詞向量 (Ghost Word Vectors)

> *將文字的靈魂封印進數學的容器*

---

## 🌟 什麼是詞向量？

**詞向量 (Word Embeddings)** 是自然語言處理中最神秘也最強大的技術之一。它將人類語言中的單詞轉換成電腦可以理解的數字向量。

想像每個單詞都是一個**幽靈**，它們在高維空間中漂浮。相似意義的幽靈會聚集在一起，形成詭異但有序的靈界結構。

---

## 🔮 為什麼叫「幽靈」詞向量？

### 👻 看不見的存在
- 單詞本身只是符號，它的「意義」像幽靈一樣無形
- 詞向量將這些無形的意義，轉化為可測量的數值

### 🕸️ 空間中的聯繫
- 幽靈們在向量空間中建立關係網
- 相似的幽靈彼此吸引，聚集在附近

### ✨ 超自然的能力
- 向量算術：「國王 - 男人 + 女人 = 皇后」
- 就像魔法公式一樣神奇！

---

## 📊 數學基礎

### 向量表示

每個單詞被表示為一個 N 維向量：

```
"南瓜" = [0.8, 0.6, -0.2, 0.4, 0.1, ...]
"萬聖節" = [0.7, 0.5, -0.1, 0.3, 0.2, ...]
"貓" = [-0.2, 0.1, 0.9, -0.3, 0.5, ...]
```

### 餘弦相似度

計算兩個幽靈的親密程度：

\[
\text{similarity}(\vec{v}_1, \vec{v}_2) = \frac{\vec{v}_1 \cdot \vec{v}_2}{\|\vec{v}_1\| \times \|\vec{v}_2\|} = \cos(\theta)
\]

- 相似度 = 1：完全相同的幽靈（同義詞）
- 相似度 = 0：毫無關係的幽靈
- 相似度 = -1：完全相反的幽靈（反義詞）

---

## 🎃 實戰範例：Word2Vec

### Skip-gram 模型

**概念**：透過上下文預測目標詞

```python
# 訓練資料範例
句子: "萬聖節的夜晚南瓜燈照亮了黑暗"

# Skip-gram 訓練對
中心詞 → 上下文詞
"南瓜燈" → "夜晚"
"南瓜燈" → "照亮"
"南瓜燈" → "了"
```

### CBOW (Continuous Bag of Words)

**概念**：透過目標詞預測上下文

```python
上下文 → 中心詞
["夜晚", "照亮", "了"] → "南瓜燈"
```

---

## 🧪 實作：從零打造幽靈詞向量

```python
import numpy as np
from collections import defaultdict

class GhostWordVector:
    """簡單的 Skip-gram 實作"""
    
    def __init__(self, vocab_size, embedding_dim=100):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 初始化權重矩陣（幽靈的初始能量）
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.01
        
        self.word2idx = {}
        self.idx2word = {}
        
    def build_vocabulary(self, corpus):
        """建立詞彙表（召喚所有幽靈）"""
        words = set()
        for sentence in corpus:
            words.update(sentence.split())
        
        for idx, word in enumerate(sorted(words)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def generate_training_data(self, corpus, window_size=2):
        """生成訓練數據（建立幽靈之間的聯繫）"""
        training_data = []
        
        for sentence in corpus:
            words = sentence.split()
            for idx, target_word in enumerate(words):
                # 取得上下文窗口
                start = max(0, idx - window_size)
                end = min(len(words), idx + window_size + 1)
                
                for context_idx in range(start, end):
                    if context_idx != idx:
                        context_word = words[context_idx]
                        training_data.append((target_word, context_word))
        
        return training_data
    
    def softmax(self, x):
        """Softmax 函數"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward(self, target_idx):
        """前向傳播"""
        # 隱藏層
        h = self.W1[target_idx]
        
        # 輸出層
        u = np.dot(h, self.W2)
        y_pred = self.softmax(u)
        
        return y_pred, h
    
    def backward(self, target_idx, context_idx, y_pred, h, learning_rate=0.01):
        """反向傳播（幽靈能量的回流）"""
        # 計算錯誤
        error = y_pred.copy()
        error[context_idx] -= 1
        
        # 更新權重
        self.W2 -= learning_rate * np.outer(h, error)
        self.W1[target_idx] -= learning_rate * np.dot(self.W2, error)
    
    def train(self, corpus, epochs=100, learning_rate=0.01):
        """訓練模型（訓練幽靈）"""
        self.build_vocabulary(corpus)
        training_data = self.generate_training_data(corpus)
        
        print(f"🔮 開始訓練 {len(self.word2idx)} 個幽靈...")
        print(f"📊 訓練樣本數：{len(training_data)}")
        
        for epoch in range(epochs):
            loss = 0
            
            for target_word, context_word in training_data:
                target_idx = self.word2idx[target_word]
                context_idx = self.word2idx[context_word]
                
                # 前向傳播
                y_pred, h = self.forward(target_idx)
                
                # 計算損失
                loss -= np.log(y_pred[context_idx] + 1e-10)
                
                # 反向傳播
                self.backward(target_idx, context_idx, y_pred, h, learning_rate)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        
        print("✅ 訓練完成！")
    
    def get_vector(self, word):
        """獲取單詞的向量"""
        if word in self.word2idx:
            idx = self.word2idx[word]
            return self.W1[idx]
        return None
    
    def similarity(self, word1, word2):
        """計算兩個單詞的相似度"""
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)
        
        if vec1 is None or vec2 is None:
            return None
        
        # 餘弦相似度
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def most_similar(self, word, top_n=5):
        """找出最相似的單詞"""
        target_vec = self.get_vector(word)
        if target_vec is None:
            return []
        
        similarities = []
        for other_word in self.word2idx:
            if other_word != word:
                sim = self.similarity(word, other_word)
                if sim is not None:
                    similarities.append((other_word, sim))
        
        # 排序並返回 top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]


# 🎃 使用範例
if __name__ == "__main__":
    # 萬聖節語料庫
    halloween_corpus = [
        "萬聖節 是 一個 詭異 的 節日",
        "南瓜 燈 在 萬聖節 很 重要",
        "鬼魂 和 殭屍 在 萬聖節 出現",
        "孩子們 在 萬聖節 收集 糖果",
        "女巫 騎著 掃帚 飛行",
        "黑貓 是 萬聖節 的 象徵",
        "吸血鬼 害怕 大蒜 和 陽光",
        "墓地 在 夜晚 很 恐怖"
    ]
    
    # 創建並訓練模型
    model = GhostWordVector(vocab_size=50, embedding_dim=50)
    model.train(halloween_corpus, epochs=100, learning_rate=0.05)
    
    # 測試相似度
    print("\n" + "="*50)
    print("🔍 測試幽靈相似度：")
    print("="*50)
    
    test_word = "萬聖節"
    similar_words = model.most_similar(test_word, top_n=5)
    
    print(f"\n與 '{test_word}' 最相似的幽靈：")
    for word, similarity in similar_words:
        print(f"  👻 {word}: {similarity:.4f}")
```

---

## 🌐 真實世界的應用

### 1. 搜尋引擎 🔍
- 理解查詢意圖
- 找出語義相關的文檔

### 2. 推薦系統 🎯
- 根據內容推薦相似商品
- "喜歡《哈利波特》的人也會喜歡..."

### 3. 機器翻譯 🌍
- 跨語言的詞向量映射
- 理解不同語言的語義對應

### 4. 情感分析 😊😢
- 識別正面/負面詞彙
- 情緒極性分類

### 5. 問答系統 💬
- 理解問題意圖
- 找出語義匹配的答案

---

## 🎓 進階技術

### GloVe (Global Vectors)
- 結合全局統計信息
- 更好的語義表示

### FastText
- 考慮子詞信息
- 處理未登錄詞（OOV）

### ELMo (Embeddings from Language Models)
- 上下文相關的詞向量
- 同一個詞在不同句子中有不同向量

### BERT
- 雙向語言模型
- 預訓練 + 微調範式

---

## 📚 學習資源

### 論文
- [Efficient Estimation of Word Representations in Vector Space (Word2Vec)](https://arxiv.org/abs/1301.3781)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
- [Enriching Word Vectors with Subword Information (FastText)](https://arxiv.org/abs/1607.04606)

### 工具庫
- **Gensim**: 最流行的 Word2Vec 實作
- **spaCy**: 預訓練的詞向量模型
- **fastText**: Facebook 開源的詞向量工具

### 視覺化工具
- [TensorFlow Embedding Projector](https://projector.tensorflow.org/)
- 在 3D 空間中探索幽靈向量！

---

## 🎃 練習題

### 基礎題
1. 實作一個簡單的詞向量相似度計算器
2. 用 Gensim 訓練 Word2Vec 模型
3. 視覺化詞向量空間（使用 t-SNE）

### 進階題
1. 實作 Skip-gram with Negative Sampling
2. 比較 Word2Vec 和 GloVe 的效果
3. 訓練中文詞向量模型

### 挑戰題
1. 實作多語言詞向量對齊
2. 用詞向量做類比推理（king - man + woman = queen）
3. 探索詞向量中的偏見問題

---

## 👻 結語

**幽靈詞向量**讓我們看到了語言的靈魂。它們：
- 🔮 將抽象的意義轉化為具體的數字
- 🕸️ 揭示了詞彙之間隱藏的關係
- ✨ 為 NLP 的黃金時代奠定了基礎

記住：*每個單詞都是一個幽靈，等待著被理解。*

---

**下一課：** [🧟 殭屍神經網絡](./🧟-zombie-neural-networks.md)

**返回目錄：** [📚 超自然 NLP 系列](./README.md)

---

<div align="center">

### 🎃 願幽靈與你同在 🎃

*"In the realm of vectors, words find their true form."*

</div>

