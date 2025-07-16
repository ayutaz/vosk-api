# Vosk日英混合音声認識モデル 実装ガイド

## 目標
- 現在の48MBモデル（vosk-model-small-ja-0.22）の精度を20%向上
- Unity/ONNX環境での動作
- 雑談会話での使用を想定

## 実装計画

### Phase 1: ドメイン適応（必須・最優先）

#### 期間・コスト
- 期間: 1-2週間
- GPU時間: 16時間（V100×1台）
- 期待精度向上: 5-10%

#### 実装手順

##### 1. データ収集（3-5日）
```bash
# 雑談会話テキストデータの収集（100MB以上）
# ソース例：
# - Common Voice日本語データセット
# - 日本語会話コーパス
# - SNS会話データ（プライバシー配慮）
# - 日英コードスイッチングサンプル

mkdir -p data/domain_adaptation
cd data/domain_adaptation

# データ収集スクリプト
python scripts/collect_conversation_data.py \
    --output corpus.txt \
    --min_size 100MB \
    --languages "ja,en" \
    --domain "casual_conversation"
```

##### 2. データ前処理（1日）
```python
# scripts/preprocess_text.py
import re
import MeCab

def preprocess_japanese_text(text):
    """日本語テキストの前処理"""
    # 正規化
    text = normalize_text(text)
    
    # 日英混合対応
    # 例: "今日のmeetingは3時から" → トークン化
    
    # カジュアル表現の正規化
    # "すげー" → "すごい"
    # "やばい" → コンテキストに応じた変換
    
    return processed_text

# 実行
python scripts/preprocess_text.py \
    --input corpus.txt \
    --output corpus_cleaned.txt
```

##### 3. 言語モデル再構築（2日）
```bash
# Kaldiフォーマットへの変換
cd $KALDI_ROOT/egs/vosk/s5

# 既存モデルのバックアップ
cp -r exp/model/graph exp/model/graph.bak

# 言語モデルの構築
utils/prepare_lang.sh \
    data/local/dict \
    "<UNK>" \
    data/local/lang \
    data/lang

# ARPAフォーマットの言語モデル作成
ngram-count \
    -text data/domain_adaptation/corpus_cleaned.txt \
    -order 3 \
    -lm data/local/lm/domain_lm.arpa

# FSTへの変換とグラフ作成
utils/format_lm.sh \
    data/lang \
    data/local/lm/domain_lm.arpa \
    data/local/dict/lexicon.txt \
    data/lang_domain

# モデルの再コンパイル（GPU使用: 8-16時間）
steps/mkgraph.sh \
    data/lang_domain \
    exp/model \
    exp/model/graph_domain
```

##### 4. 評価（1日）
```bash
# テストセットの準備
# 雑談会話の録音データ（1-2時間分）

# 評価実行
steps/decode.sh \
    --nj 4 \
    exp/model/graph_domain \
    data/test \
    exp/model/decode_domain

# CER/WERの計算
local/score.sh data/test exp/model/decode_domain

# ベースラインとの比較
python scripts/compare_results.py \
    --baseline exp/model/decode_original \
    --adapted exp/model/decode_domain
```

#### 成功基準
- CER改善: 5%以上
- 日英混合文の認識精度向上
- 推論速度の維持

### Phase 2A: DistilWhisperアプローチ（Whisper移行可能な場合）

#### 期間・コスト
- 期間: 3-4週間
- GPU時間: 200時間（A100×2台）
- 期待精度向上: 追加18-25%（合計23-35%）

#### 実装手順

##### 1. Whisper-smallの日本語ファインチューニング（1週間）
```python
# train_whisper_japanese.py
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch

# モデルとプロセッサの準備
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# 日本語データセットの準備
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "ja")

# 訓練設定
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-ja",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    num_train_epochs=3,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
)

# GPU時間: 約50時間
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
```

##### 2. 知識蒸留による軽量化（1週間）
```python
# distil_whisper.py
import torch
import torch.nn as nn
from transformers import WhisperModel

class DistilWhisperModel(nn.Module):
    def __init__(self, teacher_model, student_config):
        super().__init__()
        self.teacher = teacher_model
        self.student = WhisperModel(student_config)
        
        # 学生モデルは層数を削減
        # 244MB → 48MB目標
        
    def forward(self, inputs):
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
        
        student_outputs = self.student(**inputs)
        
        # 知識蒸留損失
        loss = self.distillation_loss(
            student_outputs, 
            teacher_outputs,
            temperature=3.0
        )
        
        return loss

# GPU時間: 約100-150時間
# 訓練実行
python distil_whisper.py \
    --teacher_model "./whisper-small-ja" \
    --output_size 48MB \
    --num_epochs 20 \
    --batch_size 32
```

##### 3. ONNX変換とUnity統合（3日）
```python
# convert_to_onnx.py
import torch
from transformers import WhisperForConditionalGeneration
import onnx

# モデルロード
model = WhisperForConditionalGeneration.from_pretrained("./distil-whisper-48mb")

# ONNX変換
torch.onnx.export(
    model,
    dummy_input,
    "whisper_48mb.onnx",
    export_params=True,
    opset_version=14,
    input_names=['input_features'],
    output_names=['logits'],
    dynamic_axes={
        'input_features': {0: 'batch_size', 1: 'sequence'},
        'logits': {0: 'batch_size', 1: 'sequence'}
    }
)

# 最適化
import onnxruntime as ort
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

### Phase 2B: BEST-RQ実装（Vosk維持の場合）

#### 期間・コスト
- 期間: 4-6週間
- GPU時間: 500時間（A100×4台）
- 期待精度向上: 追加20-30%（合計25-40%）

#### 実装手順

##### 1. ラベルなしデータ収集（1週間）
```bash
# 日本語音声データ収集（1000時間以上）
# - YouTube日本語動画
# - ポッドキャスト
# - 日英バイリンガルコンテンツ

python scripts/collect_unlabeled_audio.py \
    --sources "youtube,podcast" \
    --languages "ja,en" \
    --duration_hours 1000 \
    --output_dir data/unlabeled_audio
```

##### 2. BEST-RQ事前学習（2-3週間）
```python
# best_rq_pretrain.py
import torch
import torch.nn as nn

class BESTRQModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ランダム投影量子化器
        self.random_projection = nn.Linear(
            config.input_dim, 
            config.projection_dim, 
            bias=False
        )
        # 初期化（学習しない）
        nn.init.normal_(self.random_projection.weight)
        self.random_projection.weight.requires_grad = False
        
        # コードブック（6-8個使用）
        self.codebooks = nn.ModuleList([
            nn.Embedding(config.codebook_size, config.projection_dim)
            for _ in range(config.num_codebooks)
        ])
        
        # Conformerエンコーダー
        self.encoder = ConformerEncoder(config)
        
    def forward(self, audio_features, mask_indices):
        # マスク予測タスク
        masked_features = self.mask_audio(audio_features, mask_indices)
        
        # ランダム投影と量子化
        projected = self.random_projection(audio_features)
        quantized = self.quantize(projected)
        
        # エンコード
        predictions = self.encoder(masked_features)
        
        # 損失計算（KLダイバージェンス追加）
        loss = self.compute_loss(predictions, quantized)
        
        return loss

# GPU時間: 200-300時間
# 訓練実行
python best_rq_pretrain.py \
    --data_dir data/unlabeled_audio \
    --num_codebooks 6 \
    --batch_size 32 \
    --num_epochs 100 \
    --gpus 4
```

##### 3. ファインチューニング（1週間）
```bash
# Kaldiフォーマットでのファインチューニング
cd $KALDI_ROOT/egs/vosk/s5

# BEST-RQで学習した特徴を使用
steps/nnet3/train_dnn.py \
    --stage -10 \
    --cmd "run.pl" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 20 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 4 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --egs.dir exp/nnet3/egs \
    --use-gpu true \
    --feat-dir data/train_best_rq \
    --dir exp/nnet3/tdnn_best_rq

# GPU時間: 100時間
```

### Phase 3: 評価と最適化

#### 統合評価（1週間）
```python
# evaluate_all.py
import json
from pathlib import Path

def evaluate_model(model_path, test_set):
    """各モデルの評価"""
    results = {
        "cer": calculate_cer(model_path, test_set),
        "wer": calculate_wer(model_path, test_set),
        "rtf": calculate_rtf(model_path, test_set),  # Real-time factor
        "model_size": Path(model_path).stat().st_size / 1024 / 1024  # MB
    }
    return results

# 全モデルの比較
models = {
    "baseline": "vosk-model-small-ja-0.22",
    "domain_adapted": "exp/model/graph_domain",
    "distil_whisper": "whisper_48mb.onnx",
    "best_rq": "exp/nnet3/tdnn_best_rq"
}

results = {}
for name, path in models.items():
    results[name] = evaluate_model(path, "data/test_conversation")

# レポート生成
generate_report(results, "evaluation_report.md")
```

#### Unity統合テスト
```csharp
// UnityでのONNX Runtime統合
using Microsoft.ML.OnnxRuntime;
using System.Linq;
using UnityEngine;

public class SpeechRecognizer : MonoBehaviour
{
    private InferenceSession session;
    
    void Start()
    {
        // ONNXモデルのロード
        var modelPath = Application.streamingAssetsPath + "/whisper_48mb.onnx";
        session = new InferenceSession(modelPath);
    }
    
    public string Recognize(float[] audioData)
    {
        // 推論実行
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_features", 
                new DenseTensor<float>(audioData, new[] { 1, audioData.Length }))
        };
        
        using (var outputs = session.Run(inputs))
        {
            // デコード処理
            return DecodeOutput(outputs.First().AsEnumerable<float>());
        }
    }
}
```

## トラブルシューティング

### メモリ不足エラー
```bash
# バッチサイズを削減
--batch_size 16 → --batch_size 8

# 勾配累積を使用
--gradient_accumulation_steps 4
```

### 学習が収束しない
```python
# 学習率スケジューリング
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)
```

### 日英混合の精度が低い
```python
# 言語識別タスクを追加
multitask_loss = asr_loss + 0.1 * language_id_loss
```

## 成功の指標

| フェーズ | 目標CER改善 | 実際の改善 | 判定 |
|---------|------------|-----------|------|
| Phase 1（ドメイン適応） | 5-10% | ___ | ___ |
| Phase 2（DistilWhisper/BEST-RQ） | 追加15-25% | ___ | ___ |
| 合計 | 20-35% | ___ | ___ |

## 次のステップ

目標精度を達成できない場合：
1. Zipformerアーキテクチャへの移行（GPU 1200時間）
2. データ量の増加（特に日英混合データ）
3. アンサンブル手法の検討

## 参考リンク

- [Vosk公式ドキュメント](https://alphacephei.com/vosk/)
- [Whisper GitHub](https://github.com/openai/whisper)
- [BEST-RQ実装](https://github.com/lucasnewman/best-rq-pytorch)
- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx)