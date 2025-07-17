# ドメイン適応の詳細ガイド

## ドメイン適応とは

ドメイン適応（Domain Adaptation）は、既存の音声認識モデルを特定の用途や環境に最適化する技術です。新規にモデルを訓練するよりも**大幅に少ないリソース**で、特定ドメインでの認識精度を向上させることができます。

## なぜドメイン適応が効果的なのか

### 1. 言語モデルと音響モデルの分離
Voskは以下の2つの主要コンポーネントから構成されています：
- **音響モデル（Acoustic Model）**: 音声信号を音素に変換
- **言語モデル（Language Model）**: 音素列を単語列に変換

ドメイン適応では主に**言語モデルを更新**することで、少ないコストで大きな効果を得られます。

### 2. 雑談会話の特徴への対応
一般的な音声認識モデルは、ニュースや講演などの「きれいな」音声で訓練されています。雑談会話には以下の特徴があり、これらに対応することで精度が向上します：

```
一般的なモデル：「本日は晴天なり」
雑談会話：「今日めっちゃ晴れてるよね〜」

一般的なモデル：「会議は3時から開始します」
雑談会話：「meetingは3時からだよ」（日英混合）
```

## ドメイン適応の具体的な手法

### 1. 言語モデルの適応

#### Step 1: テキストデータの収集
```python
# scripts/collect_domain_text.py
import json
import re
from typing import List, Dict

class DomainTextCollector:
    def __init__(self):
        self.text_sources = []
        
    def collect_conversation_data(self) -> List[str]:
        """雑談会話データの収集"""
        texts = []
        
        # 1. SNSデータ（プライバシーに配慮）
        texts.extend(self.collect_sns_data())
        
        # 2. 会話コーパス
        texts.extend(self.load_conversation_corpus())
        
        # 3. チャットログ（匿名化済み）
        texts.extend(self.load_chat_logs())
        
        # 4. 日英混合文の生成
        texts.extend(self.generate_code_switch_examples())
        
        return texts
    
    def generate_code_switch_examples(self) -> List[str]:
        """日英混合文の生成"""
        templates = [
            "今日の{meeting}は{time}からです",
            "{deadline}までに{report}を提出してください",
            "その{issue}は{priority}が高いです",
            "{weekend}に{shopping}行こう"
        ]
        
        replacements = {
            "meeting": ["meeting", "ミーティング", "打ち合わせ"],
            "time": ["3時", "three o'clock", "15:00"],
            "deadline": ["deadline", "締切", "期限"],
            "report": ["report", "レポート", "報告書"],
            "issue": ["issue", "問題", "課題"],
            "priority": ["priority", "優先度", "プライオリティ"],
            "weekend": ["weekend", "週末", "土日"],
            "shopping": ["shopping", "買い物", "ショッピング"]
        }
        
        examples = []
        for template in templates:
            # 各置換パターンを生成
            examples.extend(self.expand_template(template, replacements))
        
        return examples
```

#### Step 2: テキストの前処理
```python
# scripts/preprocess_domain_text.py
import MeCab
import re
import unicodedata

class DomainTextPreprocessor:
    def __init__(self):
        self.mecab = MeCab.Tagger("-Owakati")
        self.casual_mappings = self.load_casual_mappings()
        
    def load_casual_mappings(self) -> Dict[str, str]:
        """カジュアル表現の正規化マッピング"""
        return {
            # 口語表現の正規化
            "すげー": "すごい",
            "すっげー": "すごい",
            "やべー": "やばい",
            "めっちゃ": "とても",
            "マジで": "本当に",
            "ってか": "というか",
            "つーか": "というか",
            
            # 省略形
            "してる": "している",
            "しちゃう": "してしまう",
            "しちゃった": "してしまった",
            
            # 英語の口語表現
            "gonna": "going to",
            "wanna": "want to",
            "gotta": "got to",
            
            # 日英混合での一般的な表現
            "オーケー": "OK",
            "オッケー": "OK",
        }
    
    def normalize_text(self, text: str) -> str:
        """テキストの正規化"""
        # Unicode正規化
        text = unicodedata.normalize('NFKC', text)
        
        # カジュアル表現の置換
        for casual, formal in self.casual_mappings.items():
            text = text.replace(casual, formal)
        
        # 数字の正規化
        text = self.normalize_numbers(text)
        
        # 記号の処理
        text = self.normalize_symbols(text)
        
        return text
    
    def add_code_switch_markers(self, text: str) -> str:
        """日英切り替え位置のマーキング"""
        # 言語切り替えを検出してマーカーを追加
        words = text.split()
        marked_words = []
        
        prev_lang = None
        for word in words:
            curr_lang = self.detect_language(word)
            
            if prev_lang and prev_lang != curr_lang:
                # 言語切り替えポイント
                marked_words.append("<LANG_SWITCH>")
            
            marked_words.append(word)
            prev_lang = curr_lang
        
        return " ".join(marked_words)
```

#### Step 3: 言語モデルの構築
```bash
# scripts/build_domain_lm.sh
#!/bin/bash

# 設定
KALDI_ROOT=/path/to/kaldi
DATA_DIR=data/domain_adaptation
MODEL_DIR=exp/domain_lm

# 1. 語彙の抽出
echo "Extracting vocabulary..."
python scripts/extract_vocabulary.py \
    --input $DATA_DIR/corpus_normalized.txt \
    --output $DATA_DIR/vocabulary.txt \
    --min_count 3 \
    --include_english true

# 2. 発音辞書の更新
echo "Updating pronunciation dictionary..."
python scripts/update_pronunciation.py \
    --base_dict data/local/dict/lexicon.txt \
    --new_words $DATA_DIR/vocabulary.txt \
    --output $DATA_DIR/lexicon_updated.txt

# 3. N-gramモデルの訓練
echo "Training n-gram model..."
ngram-count \
    -text $DATA_DIR/corpus_normalized.txt \
    -order 4 \
    -lm $MODEL_DIR/domain_lm.arpa \
    -kndiscount1 -kndiscount2 -kndiscount3 -kndiscount4 \
    -interpolate1 -interpolate2 -interpolate3 -interpolate4

# 4. モデルの混合（元のモデルとドメインモデル）
echo "Interpolating models..."
ngram \
    -lm data/lang_test/G.arpa \
    -mix-lm $MODEL_DIR/domain_lm.arpa \
    -lambda 0.7 \
    -write-lm $MODEL_DIR/mixed_lm.arpa

# 5. FSTへの変換
echo "Converting to FST..."
arpa2fst \
    --disambig-symbol=#0 \
    --read-symbol-table=data/lang/words.txt \
    $MODEL_DIR/mixed_lm.arpa \
    $MODEL_DIR/G.fst
```

### 2. 音響モデルの適応（オプション）

音響モデルの適応は、より多くのリソースを必要としますが、以下の場合に有効です：
- 特定の録音環境（騒音、残響など）
- 特定の話者グループ（方言、年齢層など）

```python
# scripts/acoustic_adaptation.py
import kaldi_io
import numpy as np

class AcousticModelAdapter:
    def __init__(self, base_model_path):
        self.base_model = self.load_model(base_model_path)
        
    def adapt_with_map(self, adaptation_data, tau=10):
        """MAP（Maximum A Posteriori）適応"""
        # 1. 適応データから統計量を計算
        stats = self.compute_statistics(adaptation_data)
        
        # 2. MAP推定
        adapted_params = {}
        for layer_name, params in self.base_model.items():
            if 'mean' in params:
                # 平均の適応
                adapted_params[layer_name]['mean'] = (
                    tau * params['mean'] + stats[layer_name]['sum']
                ) / (tau + stats[layer_name]['count'])
            
            if 'variance' in params:
                # 分散の適応
                adapted_params[layer_name]['variance'] = self.adapt_variance(
                    params['variance'],
                    stats[layer_name],
                    tau
                )
        
        return adapted_params
    
    def adapt_with_lhuc(self, adaptation_data):
        """LHUC（Learning Hidden Unit Contributions）適応"""
        # より軽量な適応手法
        # 各隠れ層に適応パラメータを追加
        adaptation_params = {}
        
        for layer_name in self.base_model.get_layer_names():
            # 適応データで小規模な訓練
            scale_params = self.train_lhuc_params(
                layer_name,
                adaptation_data,
                num_epochs=10
            )
            adaptation_params[layer_name] = scale_params
        
        return adaptation_params
```

## 実装例：雑談会話向けドメイン適応

### 完全な実装フロー

```python
# domain_adaptation_pipeline.py
import os
import json
from datetime import datetime

class DomainAdaptationPipeline:
    def __init__(self, base_model_path, target_domain="casual_conversation"):
        self.base_model_path = base_model_path
        self.target_domain = target_domain
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_adaptation(self):
        """ドメイン適応の実行"""
        print(f"Starting domain adaptation for {self.target_domain}")
        
        # 1. データ収集
        print("Step 1: Collecting domain data...")
        domain_texts = self.collect_domain_data()
        print(f"Collected {len(domain_texts)} text samples")
        
        # 2. データ前処理
        print("Step 2: Preprocessing...")
        processed_texts = self.preprocess_texts(domain_texts)
        
        # 3. 言語モデル構築
        print("Step 3: Building language model...")
        lm_path = self.build_language_model(processed_texts)
        
        # 4. モデル統合
        print("Step 4: Integrating models...")
        adapted_model = self.integrate_models(lm_path)
        
        # 5. 評価
        print("Step 5: Evaluating...")
        results = self.evaluate_model(adapted_model)
        
        # 6. レポート生成
        self.generate_report(results)
        
        return adapted_model, results
    
    def collect_domain_data(self):
        """ドメインデータの収集"""
        collector = DomainTextCollector()
        
        texts = []
        # 雑談会話の特徴的な表現を収集
        texts.extend(collector.collect_casual_expressions())
        texts.extend(collector.collect_filler_words())
        texts.extend(collector.collect_code_switch_patterns())
        
        # データ拡張
        augmented = self.augment_texts(texts)
        texts.extend(augmented)
        
        return texts
    
    def augment_texts(self, texts):
        """テキストデータの拡張"""
        augmented = []
        
        for text in texts:
            # バリエーション生成
            variations = [
                self.add_fillers(text),
                self.add_hesitations(text),
                self.simulate_repairs(text),
                self.add_code_switching(text)
            ]
            augmented.extend(variations)
        
        return augmented
    
    def add_fillers(self, text):
        """フィラー（えーと、あのー等）の追加"""
        fillers = ["えーと", "あのー", "まあ", "その", "なんか"]
        words = text.split()
        
        # ランダムな位置にフィラーを挿入
        import random
        if len(words) > 3 and random.random() < 0.3:
            pos = random.randint(1, len(words)-1)
            words.insert(pos, random.choice(fillers))
        
        return " ".join(words)
```

### 評価とモニタリング

```python
# scripts/evaluate_adaptation.py
class AdaptationEvaluator:
    def __init__(self, base_model, adapted_model):
        self.base_model = base_model
        self.adapted_model = adapted_model
        
    def evaluate_on_test_sets(self):
        """複数のテストセットで評価"""
        test_sets = {
            'general': 'data/test/general_test.json',
            'casual': 'data/test/casual_conversation.json',
            'code_switch': 'data/test/ja_en_mixed.json',
            'noisy': 'data/test/noisy_casual.json'
        }
        
        results = {}
        for test_name, test_path in test_sets.items():
            print(f"Evaluating on {test_name}...")
            
            base_results = self.evaluate_model(
                self.base_model, test_path
            )
            adapted_results = self.evaluate_model(
                self.adapted_model, test_path
            )
            
            improvement = (
                (base_results['cer'] - adapted_results['cer']) 
                / base_results['cer'] * 100
            )
            
            results[test_name] = {
                'base_cer': base_results['cer'],
                'adapted_cer': adapted_results['cer'],
                'improvement': improvement,
                'examples': self.get_improvement_examples(
                    base_results, adapted_results
                )
            }
        
        return results
    
    def get_improvement_examples(self, base_results, adapted_results):
        """改善例の抽出"""
        examples = []
        
        for i, (base, adapted) in enumerate(
            zip(base_results['predictions'], 
                adapted_results['predictions'])
        ):
            if base['cer'] > adapted['cer']:
                examples.append({
                    'reference': base['reference'],
                    'base_prediction': base['prediction'],
                    'adapted_prediction': adapted['prediction'],
                    'cer_improvement': base['cer'] - adapted['cer']
                })
        
        # 改善が大きい順にソート
        examples.sort(key=lambda x: x['cer_improvement'], reverse=True)
        
        return examples[:10]  # トップ10の改善例
```

## ベストプラクティス

### 1. データ品質の確保
- 実際の使用環境に近いデータを収集
- バランスの取れたデータセット（一般的な表現と特殊な表現）
- プライバシーに配慮したデータ収集

### 2. 段階的な適応
```python
# 段階的適応戦略
adaptation_stages = [
    {
        'name': 'basic_vocabulary',
        'duration': '2 days',
        'expected_improvement': '3-5%'
    },
    {
        'name': 'conversation_patterns',
        'duration': '3 days',
        'expected_improvement': '2-3%'
    },
    {
        'name': 'code_switching',
        'duration': '2 days',
        'expected_improvement': '2-4%'
    }
]
```

### 3. 継続的な改善
- ユーザーフィードバックの収集
- 定期的な再適応（月次など）
- A/Bテストによる効果測定

## まとめ

ドメイン適応は、最小限のコスト（GPU 16時間、1-2週間）で5-10%の精度向上を実現できる、最もコスト効率の高い手法です。特に雑談会話向けには：

1. **カジュアルな表現**の辞書追加
2. **日英混合パターン**の学習
3. **フィラーや言い直し**への対応

これらにより、実用的な精度向上が期待できます。