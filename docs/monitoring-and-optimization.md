# モニタリングと継続的最適化ガイド

## 実装中のモニタリング

### GPU使用状況の監視

```bash
# GPU使用状況のリアルタイム監視
nvidia-smi -l 1

# TensorBoardでの学習状況監視
tensorboard --logdir=./logs --port=6006

# 学習曲線の自動記録
python scripts/monitor_training.py \
    --experiment_name "domain_adaptation_v1" \
    --metrics "loss,cer,wer" \
    --alert_threshold 0.01
```

### 進捗管理ダッシュボード

```python
# scripts/progress_dashboard.py
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class ExperimentTracker:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.metrics_log = []
        
    def log_metrics(self, epoch, metrics):
        """メトリクスの記録"""
        self.metrics_log.append({
            'timestamp': datetime.now(),
            'epoch': epoch,
            'gpu_hours': self.calculate_gpu_hours(),
            **metrics
        })
        
    def generate_report(self):
        """進捗レポートの生成"""
        df = pd.DataFrame(self.metrics_log)
        
        # CER改善の可視化
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['cer'], label='CER')
        plt.xlabel('Epoch')
        plt.ylabel('Character Error Rate (%)')
        plt.title(f'{self.experiment_name} - CER Progress')
        plt.savefig(f'reports/{self.experiment_name}_cer.png')
        
        # GPU時間とコストの計算
        total_gpu_hours = df['gpu_hours'].sum()
        estimated_cost = total_gpu_hours * 0.5  # $0.5/hour (A100)
        
        return {
            'total_gpu_hours': total_gpu_hours,
            'estimated_cost': estimated_cost,
            'best_cer': df['cer'].min(),
            'improvement': (baseline_cer - df['cer'].min()) / baseline_cer * 100
        }
```

## データ品質の継続的改善

### 認識エラー分析

```python
# scripts/error_analysis.py
import json
from collections import Counter
import jieba
import MeCab

class ErrorAnalyzer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.mecab = MeCab.Tagger()
        
    def analyze_errors(self, test_set):
        """エラーパターンの分析"""
        errors = []
        
        for audio, reference in test_set:
            hypothesis = self.recognize(audio)
            errors.extend(self.extract_errors(reference, hypothesis))
        
        # エラーパターンの集計
        error_patterns = Counter(errors)
        
        return {
            'most_common_errors': error_patterns.most_common(20),
            'japanese_english_switch_errors': self.analyze_code_switch_errors(errors),
            'casual_speech_errors': self.analyze_casual_speech_errors(errors)
        }
    
    def analyze_code_switch_errors(self, errors):
        """日英切り替えエラーの分析"""
        switch_errors = []
        
        for error in errors:
            if self.is_code_switch_error(error):
                switch_errors.append(error)
        
        return {
            'total': len(switch_errors),
            'patterns': Counter([e['pattern'] for e in switch_errors]),
            'examples': switch_errors[:10]
        }
```

### データ拡張戦略

```python
# scripts/data_augmentation.py
import numpy as np
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

class ConversationAugmenter:
    def __init__(self):
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        ])
        
    def augment_conversation(self, audio, sample_rate):
        """雑談会話向けのデータ拡張"""
        # 基本的な拡張
        augmented = self.augment(samples=audio, sample_rate=sample_rate)
        
        # 会話特有の拡張
        # 1. 複数話者の重複をシミュレート
        if np.random.random() < 0.2:
            augmented = self.add_crosstalk(augmented)
        
        # 2. 背景雑音（カフェ、街中など）
        if np.random.random() < 0.3:
            augmented = self.add_ambient_noise(augmented)
        
        # 3. 言い淀み、フィラーの挿入
        if np.random.random() < 0.1:
            augmented = self.add_disfluencies(augmented)
        
        return augmented
```

## 性能最適化

### 推論速度の最適化

```python
# scripts/optimize_inference.py
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def optimize_for_unity(model_path):
    """Unity環境向けの最適化"""
    
    # 1. 動的量子化
    quantized_model = quantize_dynamic(
        model_path,
        "model_int8.onnx",
        weight_type=QuantType.QInt8
    )
    
    # 2. グラフ最適化
    import onnxoptimizer
    optimized_model = onnxoptimizer.optimize(quantized_model)
    
    # 3. Unity向けの最適化設定
    optimization_config = {
        'batch_size': 1,  # リアルタイム処理
        'sequence_length': 3000,  # 3秒分のオーディオ
        'enable_streaming': True
    }
    
    return optimized_model, optimization_config

# ベンチマーク実行
def benchmark_model(model_path, test_audio):
    """モデルのベンチマーク"""
    import time
    
    session = ort.InferenceSession(model_path)
    
    # ウォームアップ
    for _ in range(10):
        session.run(None, {'input': test_audio})
    
    # 計測
    times = []
    for _ in range(100):
        start = time.time()
        session.run(None, {'input': test_audio})
        times.append(time.time() - start)
    
    return {
        'mean_latency': np.mean(times),
        'p95_latency': np.percentile(times, 95),
        'rtf': np.mean(times) / (len(test_audio) / 16000)  # Real-time factor
    }
```

### メモリ使用量の最適化

```python
# scripts/memory_optimization.py
def optimize_memory_usage(model):
    """メモリ使用量の最適化"""
    
    # 1. 不要な演算の削除
    model = remove_unnecessary_ops(model)
    
    # 2. テンソルの共有
    model = share_tensors(model)
    
    # 3. ストリーミング対応
    model = enable_streaming_mode(model, chunk_size=1600)  # 100ms chunks
    
    return model

def profile_memory_usage(model_path):
    """メモリ使用量のプロファイリング"""
    import tracemalloc
    
    tracemalloc.start()
    
    # モデルロード
    session = ort.InferenceSession(model_path)
    
    # 推論実行
    for _ in range(100):
        session.run(None, test_input)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'current_memory_mb': current / 1024 / 1024,
        'peak_memory_mb': peak / 1024 / 1024
    }
```

## A/Bテストフレームワーク

```python
# scripts/ab_testing.py
class ABTestFramework:
    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b
        self.results = {'a': [], 'b': []}
        
    def run_test(self, test_set, duration_days=7):
        """A/Bテストの実行"""
        for audio, reference in test_set:
            # ランダムに振り分け
            if np.random.random() < 0.5:
                result = self.test_model_a(audio, reference)
                self.results['a'].append(result)
            else:
                result = self.test_model_b(audio, reference)
                self.results['b'].append(result)
        
        return self.analyze_results()
    
    def analyze_results(self):
        """結果の統計的分析"""
        from scipy import stats
        
        cer_a = [r['cer'] for r in self.results['a']]
        cer_b = [r['cer'] for r in self.results['b']]
        
        # t検定
        t_stat, p_value = stats.ttest_ind(cer_a, cer_b)
        
        return {
            'model_a_mean_cer': np.mean(cer_a),
            'model_b_mean_cer': np.mean(cer_b),
            'improvement': (np.mean(cer_a) - np.mean(cer_b)) / np.mean(cer_a) * 100,
            'statistically_significant': p_value < 0.05,
            'p_value': p_value
        }
```

## 継続的改善のためのパイプライン

```yaml
# .github/workflows/continuous_improvement.yml
name: Continuous Model Improvement

on:
  schedule:
    - cron: '0 0 * * 0'  # 週次実行
  workflow_dispatch:

jobs:
  evaluate_and_improve:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v2
      
      - name: Setup environment
        run: |
          pip install -r requirements.txt
          bash setup_kaldi.sh
      
      - name: Collect new data
        run: |
          python scripts/collect_new_data.py \
            --source "user_feedback" \
            --duration_hours 10
      
      - name: Analyze errors
        run: |
          python scripts/error_analysis.py \
            --model_path "models/current_best.onnx" \
            --output "reports/weekly_error_analysis.json"
      
      - name: Retrain if needed
        run: |
          python scripts/auto_retrain.py \
            --error_threshold 0.15 \
            --gpu_budget 50
      
      - name: Generate report
        run: |
          python scripts/generate_weekly_report.py \
            --metrics "cer,wer,latency,memory" \
            --output "reports/weekly_improvement.md"
```

## デプロイメントチェックリスト

### Unity統合前の確認事項

```markdown
## Pre-deployment Checklist

### モデル品質
- [ ] CER < 10% (目標値達成)
- [ ] 日英混合文の認識精度 > 85%
- [ ] 雑談会話での精度 > 90%

### 性能要件
- [ ] モデルサイズ < 48MB
- [ ] 推論速度 RTF < 0.3
- [ ] メモリ使用量 < 200MB
- [ ] 初期化時間 < 1秒

### Unity互換性
- [ ] ONNX Runtime対応確認
- [ ] iOS/Android両対応
- [ ] バッチサイズ1での動作確認

### テスト完了
- [ ] 単体テスト合格
- [ ] 統合テスト合格
- [ ] ストレステスト（24時間連続動作）
- [ ] エッジケーステスト
```

### デプロイメントスクリプト

```bash
#!/bin/bash
# deploy_to_unity.sh

# モデルの最終チェック
python scripts/final_validation.py \
    --model_path "models/final_model.onnx" \
    --checklist "deployment_checklist.yaml"

if [ $? -eq 0 ]; then
    echo "All checks passed!"
    
    # Unity用にパッケージング
    python scripts/package_for_unity.py \
        --model "models/final_model.onnx" \
        --config "configs/unity_config.json" \
        --output "unity_package/"
    
    # ドキュメント生成
    python scripts/generate_unity_docs.py \
        --output "docs/unity_integration.md"
    
    echo "Deployment package ready at: unity_package/"
else
    echo "Deployment checks failed. Please review the errors."
    exit 1
fi
```

## トラブルシューティングガイド

### よくある問題と解決策

| 問題 | 原因 | 解決策 |
|------|------|--------|
| 学習が収束しない | 学習率が高すぎる | 学習率を1/10に減らす |
| 日英切り替えで精度低下 | データ不足 | コードスイッチングデータを追加 |
| メモリ不足エラー | バッチサイズが大きい | バッチサイズを半分に |
| 推論が遅い | 量子化されていない | INT8量子化を適用 |

### デバッグツール

```python
# scripts/debug_tools.py
def debug_recognition_error(audio_file, expected_text):
    """認識エラーのデバッグ"""
    # 1. 音声の可視化
    visualize_audio(audio_file)
    
    # 2. 特徴量の確認
    features = extract_features(audio_file)
    plot_features(features)
    
    # 3. モデルの各層の出力確認
    layer_outputs = get_intermediate_outputs(model, audio_file)
    analyze_layer_outputs(layer_outputs)
    
    # 4. ビーム探索の経路確認
    beam_paths = trace_beam_search(model, audio_file)
    visualize_beam_paths(beam_paths)
    
    return generate_debug_report()
```

これで実装ガイドの全体が完成しました。このドキュメントに従って段階的に実装を進めることで、効率的に目標を達成できます。