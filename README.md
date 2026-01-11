# Multi-Agent Swarm Intelligence Simulation

5 kara robotunun 2D grid ortamında MADDPG (Multi-Agent Deep Deterministic Policy Gradient) ve GNN (Graph Neural Network) kullanarak kollektif keşif yapmasını simüle eden bir proje.

## Özellikler

- **Multi-Agent Reinforcement Learning**: MADDPG algoritması ile 5 robot koordineli öğrenme
- **Graph Neural Network**: Robotlar arası iletişim ve bilgi paylaşımı
- **2D Grid Exploration**: Keşif görevi ile alan kapsama optimizasyonu
- **Real-time Visualization**: Pygame ile canlı görselleştirme
- **TensorBoard Logging**: Training metrikleri ve grafikleri
- **Configurable**: YAML dosyası ile tüm parametreler ayarlanabilir

## Kurulum

### 1. Virtual Environment Oluşturma

```bash
# Python 3.13 gerekli (pyproject.toml'de belirtildiği gibi)
python3.13 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# veya
.venv\Scripts\activate  # Windows
```

### 2. Bağımlılıkları Yükleme

```bash
# uv kullanarak (önerilen)
uv pip install -e .

# veya pip ile
pip install -e .
```

## Kullanım

### Training (Model Eğitimi)

Varsayılan konfigürasyon ile training başlat:

```bash
python main.py --mode train
```

Özel parametrelerle:

```bash
python main.py --mode train \
    --config configs/default_config.yaml \
    --episodes 2000 \
    --device cuda \
    --seed 42
```

Training sırasında görselleştirme olmadan (daha hızlı):

```bash
python main.py --mode train --no-render
```

### Evaluation (Model Test)

Eğitilmiş modeli test et:

```bash
python main.py --mode eval \
    --model trained_models/maddpg_final.pt \
    --episodes 100
```

### Demo (Tek Episode Görselleştirme)

Modelin performansını tek bir episode'da izle:

```bash
python main.py --mode demo --model trained_models/maddpg_final.pt
```

## Proje Yapısı

```
swarm_intelligence_simulation/
├── src/
│   ├── environment/
│   │   └── exploration_env.py      # PettingZoo tabanlı 2D grid environment
│   ├── agents/
│   │   └── maddpg.py               # MADDPG agent implementasyonu
│   ├── models/
│   │   └── gnn.py                  # Graph Neural Network modeli
│   ├── training/
│   │   └── trainer.py              # Training loop
│   ├── evaluation/
│   │   └── evaluator.py            # Evaluation ve metrikler
│   └── utils/
│       ├── replay_buffer.py        # Experience replay buffer
│       └── helpers.py              # Yardımcı fonksiyonlar
├── configs/
│   └── default_config.yaml         # Konfigürasyon dosyası
├── trained_models/                 # Kaydedilen model checkpoints
├── results/
│   ├── logs/                       # TensorBoard logs
│   ├── plots/                      # Evaluation grafikleri
│   └── videos/                     # Kayıtlı videolar (opsiyonel)
├── main.py                         # Ana giriş noktası
├── pyproject.toml                  # Proje bağımlılıkları
└── README.md                       # Bu dosya
```

## Konfigürasyon

`configs/default_config.yaml` dosyasında tüm parametreler düzenlenebilir:

### Environment Ayarları
- `grid_size`: Grid boyutu (25x25)
- `num_agents`: Robot sayısı (5)
- `num_obstacles`: Engel sayısı (15)
- `communication_range`: İletişim mesafesi (5.0)
- `max_steps`: Episode başına maksimum adım (200)

### Model Ayarları
- `actor_hidden_dims`: Actor network katmanları [128, 128]
- `critic_hidden_dims`: Critic network katmanları [256, 256]
- `gnn_hidden_dim`: GNN gizli katman boyutu (64)
- `gnn_num_layers`: GNN katman sayısı (2)
- `actor_lr`: Actor learning rate (0.0001)
- `critic_lr`: Critic learning rate (0.001)

### Training Ayarları
- `num_episodes`: Toplam episode sayısı (2000)
- `batch_size`: Batch boyutu (256)
- `buffer_size`: Replay buffer kapasitesi (100000)
- `epsilon_start/end/decay`: Exploration parametreleri

### Reward Ayarları
- `exploration`: Yeni alan keşfi ödülü (1.0)
- `collision`: Çarpışma cezası (-0.5)
- `collective_bonus`: Kollektif keşif bonusu (2.0)

## Algoritma Detayları

### MADDPG (Multi-Agent DDPG)
- **Actor-Critic Architecture**: Her robot için ayrı actor ve critic network
- **Centralized Training, Decentralized Execution**: Training sırasında tüm bilgi paylaşılır, execution'da her robot bağımsız
- **Experience Replay**: Geçmiş deneyimlerden öğrenme
- **Soft Target Updates**: Kararlı öğrenme için target network'ler

### GNN (Graph Neural Network)
- **Graph Attention Network (GAT)**: Multi-head attention mekanizması
- **Dynamic Graph**: Communication range'e göre dinamik komşuluk
- **Message Passing**: Robotlar arası bilgi akışı
- **Residual Connections**: Daha derin network'ler için

### Environment
- **Grid-based 2D World**: Discrete action space (Up, Down, Left, Right, Stay)
- **Partial Observability**: Her robot sadece kendi view range'ini görür
- **Communication**: Belirli mesafedeki robotlarla iletişim
- **Reward Shaping**: Keşif, çarpışmadan kaçınma, kollektif davranış

## TensorBoard ile İzleme

Training sırasında metrikleri izlemek için:

```bash
tensorboard --logdir results/logs
```

Browser'da `http://localhost:6006` adresini açın.

## Beklenen Sonuçlar

Başarılı bir training sonrasında:
- **Exploration Rate**: %70-90 alan kapsama
- **Collision Rate**: Minimum çarpışma
- **Coordination**: Robotlar birbirlerini tamamlayıcı şekilde hareket eder
- **Communication**: Komşu robotlarla bilgi paylaşımı

## Troubleshooting

### CUDA/MPS hataları
```bash
# CPU'da çalıştır
python main.py --mode train --device cpu
```

### Pygame görselleştirme çok yavaş
```bash
# Render'ı kapat
python main.py --mode train --no-render
```

### Memory hatası
```bash
# Batch size'ı küçült (config dosyasında)
batch_size: 128  # varsayılan 256
buffer_size: 50000  # varsayılan 100000
```

## Gelişmiş Özellikler

### Hyperparameter Tuning
Config dosyasını kopyalayıp farklı parametrelerle deneyler yapabilirsiniz:

```bash
cp configs/default_config.yaml configs/experiment_1.yaml
# experiment_1.yaml'ı düzenle
python main.py --mode train --config configs/experiment_1.yaml
```

### Checkpoint'lerden Devam Etme
Model checkpoint'leri otomatik olarak kaydedilir (`save_frequency: 100`). Training'i kesip daha sonra devam ettirebilirsiniz.

