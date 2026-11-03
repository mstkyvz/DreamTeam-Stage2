# DreamTeam — Stage 2: Sorgu / Ürün Alaka Sınıflandırması

Bir arama sorgusu ile bir ürün arasındaki alakayı **üç sınıfa** ayıran, tek GPU üzerinde çalışan,
Docker ile paketlenmiş yarışma çözümü.

<p>
  <img alt="macro-F1" src="https://img.shields.io/badge/macro--F1-0.9131-brightgreen">
  <img alt="runtime" src="https://img.shields.io/badge/90k%20sat%C4%B1r-46.9%20dk-blue">
  <img alt="gpu" src="https://img.shields.io/badge/GPU-A100%2040GB-76B900">
  <img alt="model" src="https://img.shields.io/badge/model-Gemma--4--31B%20INT8-orange">
  <img alt="vllm" src="https://img.shields.io/badge/vLLM-0.28.0-purple">
</p>

---

## Öne Çıkanlar

- **Tek model, tek token:** `Gemma-4-31B-it-INT8` modeline rubrik ve sabit few-shot örnekleri
  verilir, yanıt `{"etiket": "` ile önceden doldurulur ve **yalnızca bir token** logprob'larıyla
  çözülür. JSON ayrıştırma yok, bozuk çıktı riski yok.
- **Kalibre edilmiş okuma:** Üç sınıflı olasılık vektörü üzerinde global IPF (λ=0.5) uygulanır,
  etiket `argmax` ile seçilir.
- **Hız:** 90.000 satır **46.9 dakikada** (limit 150 dk), ~33 satır/sn.
- **Açıklanabilirlik:** FastAPI tabanlı arayüz; posterior, alan bazlı occlusion attribution ve
  routing marjını aynı tahmin yolundan gösterir.
- **Kısıtlı ortama uygun:** `--read-only`, `--network none`, `--cap-drop ALL`, rastgele UID ile
  çalışır. Tüm cache'ler `/tmp`'ye yönlendirilmiştir.
- **Test edilebilir mimari:** Ports & adapters; tüm orkestrasyon GPU olmadan mock engine ile
  test edilir.

## Sonuçlar

90.000 satırlık hold-out set, A100-SXM4-40GB:

| konfigürasyon | macro-F1 | süre |
| --- | ---: | ---: |
| cascade @ %5 yönlendirme | 0.8741 | 24.8 dk |
| cascade @ %10 yönlendirme | 0.8798 | 28.6 dk |
| cascade @ %15 yönlendirme | 0.8903 | 29.4 dk |
| cascade @ %20 yönlendirme | ≤ 0.8935 | 32.8 dk |
| **yalnızca champion (teslim edilen)** | **0.9131** | **46.9 dk** |

Cascade her oranda daha hızlı ama daha düşük skorlu; süre limiti bol olduğu için teslim edilen
imaj yalnızca 31B champion modelini içerir.

## Repo Yapısı

```
.
├── DreamTeam_Stage2_Teknik_Rapor.pdf   # teknik rapor
└── DreamTeam_docker/
    ├── Dockerfile
    ├── build_docker.sh                 # imajı derler + imaj içi smoke test
    ├── fetch_models.sh                 # model ağırlıklarını indirir (~32 GB, HF_TOKEN gerekir)
    ├── run_solution.sh                 # tahmin modu (organizatör scripti)
    ├── run_explainability.sh           # açıklanabilirlik arayüzü + ngrok (organizatör scripti)
    ├── smoke_test.sh                   # mock engine ile uçtan uca test
    ├── requirements.txt
    ├── src/
    │   ├── main.py                     # giriş noktası: main.py INPUT.csv OUTPUT.csv
    │   ├── app/                        # config, contracts, errors, wiring (composition root)
    │   ├── services/                   # cascade, routing, readout, io, explain
    │   ├── adapters/                   # vLLM ve mock inference engine
    │   ├── api/                        # açıklanabilirlik HTTP servisi ve UI
    │   ├── core/                       # ölçümlerin yapıldığı sayısal yol (değiştirilmeden)
    │   └── assets/                     # sabit few-shot bloğu
    └── tests/
```

## Hızlı Başlangıç

```bash
cd DreamTeam_docker

# 1) Model ağırlıklarını indir (bir kez; HF_TOKEN ve internet gerekir)
HF_TOKEN="hf_..." ./fetch_models.sh

# 2) İmajı derle
CONTAINER_NAME="DreamTeam_solution" ./build_docker.sh

# 3) Tahmin üret
CONTAINER_NAME="DreamTeam_solution" \
  ./run_solution.sh /abs/path/input.csv /abs/path/submission.csv

# 4) Açıklanabilirlik arayüzü
NGROK_AUTHTOKEN="..." CONTAINER_NAME="DreamTeam_solution" \
  ./run_explainability.sh /abs/path/public_test.csv
```

**Girdi:** `row_id,query,title,category,brand,gender,age_group,attributes`
**Çıktı:** `row_id,real_label`

### GPU olmadan test

```bash
cd DreamTeam_docker
python3 -m pytest tests/ -q   # GPU gerekmez, mock engine
./smoke_test.sh               # main.py uçtan uca, mock engine ile
```

## Teknik Detaylar

| öğe | değer |
| --- | --- |
| Base imaj | `nvidia/cuda:13.0.0-devel-ubuntu24.04` |
| Serving | `vllm==0.28.0`, `flashinfer-python==0.6.16.post3` |
| Model | `lokeshe09/gemma-4-31B-it-INT8` (W8A8, SM80'de native) |
| Attention | FlashInfer |
| `max_model_len` | 5250 |
| `gpu_memory_utilization` | 0.94 |
| Batching | `max_num_seqs=256`, `max_num_batched_tokens=4096`, prefix caching açık |
| Decoding | greedy, `temperature=0`, `max_tokens=1`, `logprobs=48`, `seed=0` |
| Tokenizasyon | ortak prefix bir kez tokenize edilip splice edilir: 588 sn → 27 sn, 90k/90k bit-identical |

Ölçümler, tasarım kararları, kısıtlar ve bilinen sınırlamalar için ayrıntılı dokümantasyon:
**[DreamTeam_docker/README.md](DreamTeam_docker/README.md)** ve
**[Teknik Rapor (PDF)](DreamTeam_Stage2_Teknik_Rapor.pdf)**.

## Lisans ve Kullanılan Varlıklar

- Model: `lokeshe09/gemma-4-31B-it-INT8` — [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- Eğitim ve değerlendirme verisi yarışma koşullarına tabidir ve bu repoda yer almaz.
