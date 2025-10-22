# 💀 骷髏程式設計 (Skeleton Programming)

> *先搭骨架，再填血肉 - 從結構開始的編程藝術*

---

## 🦴 什麼是骷髏程式設計？

**骷髏程式設計 (Skeleton Programming)** 是一種基於高層程序結構和假代碼（dummy code）的編程方法論。

就像骷髏是身體的框架，骷髏程式設計強調：
- 💀 **先建立程序的骨架結構**
- 🦴 **使用佔位符和假代碼**
- 🧠 **逐步填充實際實現**
- ⚡ **保持結構清晰可見**

---

## 👻 為什麼叫「骷髏」？

### 骷髏的特性 vs 程式骨架

| 骷髏特性 | 程式設計對應 |
|---------|-------------|
| 💀 **支撐整個身體** | 程序的主要結構和架構 |
| 🦴 **206 塊骨頭** | 模組化的函數和類別 |
| 🔗 **關節連接** | 函數之間的介面和調用 |
| 📐 **對稱平衡** | 代碼的邏輯組織 |
| ⚪ **堅固但輕盈** | 簡潔但完整的框架 |

---

## 🎯 核心原則

### 1. 自上而下設計 (Top-Down Design)

```
整體程序
    ↓
主要模組
    ↓
子模組
    ↓
具體函數
```

**先思考整體結構，再填充細節**

### 2. 漸進式開發 (Incremental Development)

```
骷髏階段 → 肌肉階段 → 皮膚階段 → 完整程序
  ↓           ↓           ↓           ↓
結構框架    核心邏輯    細節實現    優化完善
```

### 3. 佔位符使用 (Placeholder Usage)

```python
# 骷髏階段：使用 pass 或返回假數據
def calculate_complex_algorithm(data):
    pass  # TODO: 實作複雜演算法

def fetch_database_records():
    return []  # 假數據，稍後實作
```

---

## 🦴 骷髏程式設計實戰

### 範例 1：建立一個萬聖節活動管理系統

#### 階段 1：骷髏骨架 💀

```python
class HalloweenEventManager:
    """萬聖節活動管理系統的骷髏"""
    
    def __init__(self):
        """初始化系統"""
        pass
    
    def register_participant(self, name, costume):
        """註冊參加者"""
        pass
    
    def assign_candy_quota(self, participant_id):
        """分配糖果配額"""
        pass
    
    def calculate_scare_score(self, costume_type):
        """計算恐怖指數"""
        pass
    
    def generate_event_report(self):
        """生成活動報告"""
        pass
    
    def send_notifications(self, message):
        """發送通知"""
        pass


# 骷髏測試
if __name__ == "__main__":
    manager = HalloweenEventManager()
    print("💀 骷髏系統已建立")
```

**優點：**
- ✅ 清楚看到系統有哪些功能
- ✅ 可以開始設計介面
- ✅ 團隊成員可以並行開發
- ✅ 易於審查架構設計

---

#### 階段 2：添加肌肉（基本邏輯）🦴

```python
class HalloweenEventManager:
    """萬聖節活動管理系統 - 添加基本邏輯"""
    
    def __init__(self):
        """初始化系統"""
        self.participants = {}  # 參加者字典
        self.candy_inventory = 1000  # 糖果庫存
        self.event_log = []  # 活動日誌
    
    def register_participant(self, name, costume):
        """註冊參加者"""
        participant_id = len(self.participants) + 1
        self.participants[participant_id] = {
            'name': name,
            'costume': costume,
            'candy_received': 0,
            'scare_score': self.calculate_scare_score(costume)
        }
        self.event_log.append(f"👻 {name} 註冊成功（裝扮：{costume}）")
        return participant_id
    
    def assign_candy_quota(self, participant_id):
        """分配糖果配額"""
        if participant_id not in self.participants:
            return None
        
        # 基於恐怖指數分配糖果
        participant = self.participants[participant_id]
        scare_score = participant['scare_score']
        candy_amount = min(scare_score * 10, self.candy_inventory)
        
        participant['candy_received'] = candy_amount
        self.candy_inventory -= candy_amount
        
        return candy_amount
    
    def calculate_scare_score(self, costume_type):
        """計算恐怖指數"""
        scare_levels = {
            '殭屍': 8,
            '吸血鬼': 9,
            '鬼魂': 7,
            '女巫': 6,
            '南瓜': 3,
            '公主': 2
        }
        return scare_levels.get(costume_type, 5)
    
    def generate_event_report(self):
        """生成活動報告"""
        report = "🎃 萬聖節活動報告 🎃\n"
        report += "=" * 40 + "\n"
        report += f"參加人數：{len(self.participants)}\n"
        report += f"剩餘糖果：{self.candy_inventory}\n"
        report += "\n參加者列表：\n"
        
        for pid, info in self.participants.items():
            report += f"  {info['name']} - {info['costume']} - "
            report += f"糖果：{info['candy_received']} - "
            report += f"恐怖度：{info['scare_score']}/10\n"
        
        return report
    
    def send_notifications(self, message):
        """發送通知"""
        print(f"📢 通知：{message}")
        self.event_log.append(f"📢 {message}")


# 測試系統
if __name__ == "__main__":
    manager = HalloweenEventManager()
    
    # 註冊參加者
    id1 = manager.register_participant("小明", "殭屍")
    id2 = manager.register_participant("小華", "吸血鬼")
    id3 = manager.register_participant("小美", "南瓜")
    
    # 分配糖果
    for pid in [id1, id2, id3]:
        candy = manager.assign_candy_quota(pid)
        manager.send_notifications(f"參加者 {pid} 獲得 {candy} 顆糖果")
    
    # 生成報告
    print("\n" + manager.generate_event_report())
```

---

#### 階段 3：添加皮膚（細節優化）✨

```python
import datetime
import json
from typing import Dict, List, Optional

class HalloweenEventManager:
    """萬聖節活動管理系統 - 完整版本"""
    
    def __init__(self, initial_candy: int = 1000):
        """
        初始化系統
        
        Args:
            initial_candy: 初始糖果數量
        """
        self.participants: Dict[int, dict] = {}
        self.candy_inventory = initial_candy
        self.event_log: List[str] = []
        self.start_time = datetime.datetime.now()
    
    def register_participant(self, name: str, costume: str, 
                           age: Optional[int] = None) -> int:
        """
        註冊參加者
        
        Args:
            name: 參加者姓名
            costume: 裝扮類型
            age: 年齡（可選）
        
        Returns:
            參加者ID
        
        Raises:
            ValueError: 如果姓名為空
        """
        if not name:
            raise ValueError("姓名不能為空")
        
        participant_id = len(self.participants) + 1
        self.participants[participant_id] = {
            'name': name,
            'costume': costume,
            'age': age,
            'candy_received': 0,
            'scare_score': self.calculate_scare_score(costume),
            'registration_time': datetime.datetime.now().isoformat()
        }
        
        log_msg = f"👻 {name} 註冊成功（裝扮：{costume}）"
        if age:
            log_msg += f" - 年齡：{age}"
        self.event_log.append(log_msg)
        
        return participant_id
    
    def assign_candy_quota(self, participant_id: int, 
                          bonus_multiplier: float = 1.0) -> Optional[int]:
        """
        分配糖果配額
        
        Args:
            participant_id: 參加者ID
            bonus_multiplier: 獎勵倍數
        
        Returns:
            分配的糖果數量，如果失敗返回 None
        """
        if participant_id not in self.participants:
            self.event_log.append(f"❌ 找不到參加者 {participant_id}")
            return None
        
        participant = self.participants[participant_id]
        scare_score = participant['scare_score']
        
        # 計算基礎糖果量
        base_candy = scare_score * 10
        
        # 兒童加成
        if participant.get('age') and participant['age'] < 12:
            base_candy = int(base_candy * 1.5)
        
        # 應用獎勵倍數
        candy_amount = int(base_candy * bonus_multiplier)
        candy_amount = min(candy_amount, self.candy_inventory)
        
        participant['candy_received'] = candy_amount
        self.candy_inventory -= candy_amount
        
        self.event_log.append(
            f"🍬 {participant['name']} 獲得 {candy_amount} 顆糖果"
        )
        
        return candy_amount
    
    def calculate_scare_score(self, costume_type: str) -> int:
        """
        計算恐怖指數
        
        Args:
            costume_type: 裝扮類型
        
        Returns:
            恐怖指數 (1-10)
        """
        scare_levels = {
            '殭屍': 8,
            '吸血鬼': 9,
            '鬼魂': 7,
            '女巫': 6,
            '骷髏': 8,
            '狼人': 9,
            '南瓜': 3,
            '公主': 2,
            '超級英雄': 4
        }
        return scare_levels.get(costume_type, 5)
    
    def generate_event_report(self, format: str = 'text') -> str:
        """
        生成活動報告
        
        Args:
            format: 報告格式 ('text' 或 'json')
        
        Returns:
            格式化的報告字符串
        """
        if format == 'json':
            return self._generate_json_report()
        
        report = []
        report.append("🎃 萬聖節活動報告 🎃")
        report.append("=" * 50)
        report.append(f"活動開始時間：{self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"參加人數：{len(self.participants)}")
        report.append(f"已發放糖果：{1000 - self.candy_inventory}")
        report.append(f"剩餘糖果：{self.candy_inventory}")
        report.append(f"平均恐怖指數：{self._calculate_average_scare():.1f}/10")
        report.append("\n" + "🎭 參加者詳細資訊 🎭")
        report.append("-" * 50)
        
        for pid, info in sorted(self.participants.items()):
            report.append(
                f"  [{pid}] {info['name']:10s} | "
                f"{info['costume']:8s} | "
                f"糖果: {info['candy_received']:3d} | "
                f"恐怖度: {info['scare_score']}/10"
            )
        
        report.append("\n" + "📜 活動日誌 📜")
        report.append("-" * 50)
        for log in self.event_log[-10:]:  # 顯示最後10條
            report.append(f"  {log}")
        
        return "\n".join(report)
    
    def _generate_json_report(self) -> str:
        """生成 JSON 格式報告"""
        report_data = {
            'event_info': {
                'start_time': self.start_time.isoformat(),
                'total_participants': len(self.participants),
                'candy_distributed': 1000 - self.candy_inventory,
                'candy_remaining': self.candy_inventory,
                'average_scare_score': self._calculate_average_scare()
            },
            'participants': list(self.participants.values()),
            'event_log': self.event_log
        }
        return json.dumps(report_data, indent=2, ensure_ascii=False)
    
    def _calculate_average_scare(self) -> float:
        """計算平均恐怖指數"""
        if not self.participants:
            return 0.0
        total = sum(p['scare_score'] for p in self.participants.values())
        return total / len(self.participants)
    
    def send_notifications(self, message: str, 
                          target: Optional[int] = None) -> None:
        """
        發送通知
        
        Args:
            message: 通知訊息
            target: 目標參加者ID（None 表示廣播）
        """
        if target:
            if target in self.participants:
                name = self.participants[target]['name']
                print(f"📧 [{name}] {message}")
                self.event_log.append(f"📧 通知 {name}: {message}")
        else:
            print(f"📢 [廣播] {message}")
            self.event_log.append(f"📢 廣播: {message}")
    
    def get_top_scariest(self, n: int = 3) -> List[dict]:
        """
        獲取最恐怖的前 N 個裝扮
        
        Args:
            n: 返回數量
        
        Returns:
            排序後的參加者列表
        """
        sorted_participants = sorted(
            self.participants.values(),
            key=lambda x: x['scare_score'],
            reverse=True
        )
        return sorted_participants[:n]
    
    def export_data(self, filename: str) -> None:
        """匯出數據到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.generate_event_report(format='json'))
        print(f"💾 數據已匯出到 {filename}")


# 🎃 完整測試
if __name__ == "__main__":
    print("💀 啟動萬聖節活動管理系統\n")
    
    manager = HalloweenEventManager(initial_candy=500)
    
    # 批量註冊
    participants_data = [
        ("小明", "殭屍", 10),
        ("小華", "吸血鬼", 12),
        ("小美", "南瓜", 8),
        ("阿強", "狼人", 15),
        ("小芳", "女巫", 11),
    ]
    
    print("👻 註冊參加者...\n")
    for name, costume, age in participants_data:
        pid = manager.register_participant(name, costume, age)
        candy = manager.assign_candy_quota(pid)
        print(f"✅ {name} 註冊成功，獲得 {candy} 顆糖果")
    
    print("\n" + "="*50 + "\n")
    
    # 顯示最恐怖的裝扮
    print("🏆 最恐怖裝扮排行榜：\n")
    for i, participant in enumerate(manager.get_top_scariest(3), 1):
        print(f"  {i}. {participant['name']} - {participant['costume']} "
              f"(恐怖度: {participant['scare_score']}/10)")
    
    print("\n" + "="*50 + "\n")
    
    # 生成完整報告
    print(manager.generate_event_report())
    
    # 匯出數據
    print("\n" + "="*50 + "\n")
    manager.export_data("halloween_event.json")
```

---

## 🎓 骷髏程式設計的優勢

### ✅ 優點

1. **清晰的結構視野**
   - 一眼看出程序的整體架構
   - 易於團隊理解和溝通

2. **並行開發**
   - 團隊成員可以同時填充不同模組
   - 介面定義清楚，減少衝突

3. **漸進式測試**
   - 每個階段都可以測試
   - 及早發現設計問題

4. **靈活調整**
   - 在實作前修改設計成本低
   - 易於重構骨架

5. **文檔化**
   - 骨架本身就是最好的文檔
   - 新人容易上手

### ⚠️ 注意事項

1. **避免過度設計**
   - 不要建立過於複雜的骨架
   - YAGNI 原則：You Aren't Gonna Need It

2. **保持骨架更新**
   - 實作時發現問題要及時調整骨架
   - 避免骨架與實際代碼脫節

3. **適度使用佔位符**
   - 不要留太多 TODO
   - 設定明確的填充計劃

---

## 🛠️ 骷髏程式設計工具

### Python Stub 生成工具

```python
# 使用 stubgen 生成骨架
# pip install mypy
# stubgen module_name.py
```

### IDE 支持

- **PyCharm**: 自動生成方法骨架
- **VS Code**: Code Snippets
- **Vim**: UltiSnips

### 文檔工具

```python
# Sphinx 文檔生成
"""
.. autoclass:: HalloweenEventManager
   :members:
   :undoc-members:
   :show-inheritance:
"""
```

---

## 📚 實戰練習

### 練習 1：建立鬼屋遊戲骨架

**需求：**
- 多個房間的鬼屋探險遊戲
- 玩家可以移動、收集物品、與 NPC 互動
- 有戰鬥系統和物品欄

**任務：** 
1. 設計骨架結構
2. 定義主要類別和方法
3. 撰寫 docstring

### 練習 2：實作萬聖節糖果交易系統

**需求：**
- 孩子們可以交換糖果
- 不同糖果有不同價值
- 紀錄交易歷史

**任務：**
1. 先建立骨架
2. 逐步填充邏輯
3. 添加錯誤處理

---

## 🎃 最佳實踐

### 1. 命名規範
```python
# 好的骨架命名
class GhostHunter:
    def search_haunted_location(self, location: str) -> List[Ghost]:
        pass
    
    def capture_ghost(self, ghost: Ghost) -> bool:
        pass

# 不好的命名
class GH:
    def sh(self, l):
        pass
```

### 2. 文檔字符串
```python
def calculate_candy_distribution(
    participants: List[Participant],
    total_candy: int,
    fairness_mode: str = 'equal'
) -> Dict[int, int]:
    """
    計算糖果分配方案
    
    Args:
        participants: 參加者列表
        total_candy: 總糖果數
        fairness_mode: 分配模式 ('equal', 'merit', 'random')
    
    Returns:
        字典，鍵為參加者ID，值為分配的糖果數
    
    Raises:
        ValueError: 如果糖果數量不足
    
    Examples:
        >>> calculate_candy_distribution([p1, p2], 100, 'equal')
        {1: 50, 2: 50}
    """
    pass
```

### 3. 類型註解
```python
from typing import List, Dict, Optional, Union

class HalloweenParty:
    guests: List[str]
    candy_stash: Dict[str, int]
    
    def invite_guest(self, name: str, bring_candy: Optional[int] = None) -> bool:
        pass
```

---

## 💀 結語

**骷髏程式設計**教會我們：
- 🦴 結構先於細節
- 💀 框架決定質量
- 🧠 思考先於編碼
- ✨ 漸進式開發

記住：*好的骨架支撐強壯的程序，就像骷髏支撐整個身體！*

---

**相關主題：**
- [🧟 殭屍程式碼重構](./🧟-zombie-code-refactoring.md)
- [👻 幽靈詞向量](./👻-ghost-word-vectors.md)
- [🎃 萬聖節設計模式](./🎃-halloween-design-patterns.md)

---

<div align="center">

### 💀 願你的代碼骨架堅固，邏輯清晰！💀

*"First the skeleton, then the flesh, finally the soul."*

</div>

