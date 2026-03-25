# TruthRAG

### Semantik Parçalama, Yeniden Sıralama ve Değerlendirme ile Kendini Düzelten RAG Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangGraph-0.2-orange" alt="LangGraph">
  <img src="https://img.shields.io/badge/Ollama-lokal_LLM-black?logo=ollama" alt="Ollama">
  <img src="https://img.shields.io/badge/Qdrant-vektör_DB-dc382c" alt="Qdrant">
  <img src="https://img.shields.io/badge/Streamlit-arayüz-ff4b4b?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Docker-konteyner-2496ED?logo=docker&logoColor=white" alt="Docker">
</p>

**[English documentation](README.md)**

---

Standart RAG pipeline'ları dokümanlardan parçalar çeker, LLM'e verir ve çıkan cevabı olduğu gibi sunar. Parça alakasız mı, model bir şey uyduruyor mu -- kimse kontrol etmez.

TruthRAG farklı çalışıyor. Her parçayı kullanmadan önce puanlıyor, üretilen cevabı kaynaklarla karşılaştırıyor, tutarsızlık bulursa baştan üretiyor. Yerel dokümanlarda cevap yoksa web'e düşüyor.

Bunun ötesinde, dokümanları anlam sınırlarından bölen semantik parçalama, vektör benzerliğinin ötesine geçen cross-encoder yeniden sıralama ve tek bir sorunun yakalayamayacağı içerikleri bulmak için sorgu genişletme kullanıyor.

Her şey lokal çalışıyor -- API anahtarı yok, bulut bağımlılığı yok, veriniz makinenizden çıkmıyor.

---

## Pipeline

```
Soru girer
    |
    v
 1. SORGU GENİŞLETME -- LLM alternatif ifadeler üretir, daha geniş arama yapar
    |
    v
 2. HİBRİT ARAMA -- vektör benzerlik (%70) + BM25 anahtar kelime (%30)
    |                  reciprocal rank fusion ile birleştirilir
    v
 3. CROSS-ENCODER YENİDEN SIRALAMA -- ms-marco-MiniLM adayları yeniden puanlar
    |
    v
 4. PUANLAMA -- LLM her parçayı sorar: "Bu ne kadar ilgili?" (0.0-1.0)
    |
    +-- hepsi düşük? --> WEB ARAMA (DuckDuckGo + Crawl4AI) --> tekrar puanla
    |
    v
 5. YANIT ÜRET -- Geçen parçalardan [Source N] atıflı cevap
    |
    v
 6. HALÜSİNASYON KONTROLÜ -- Yanıt kaynaklara dayalı mı?
    |
    +-- tutarsız? --> tekrar üret (maks 2 deneme)
    |
    v
 Sonuç: yanıt + atıflar + güven skoru + metadata
```

Akış bir **LangGraph StateGraph** -- her adım bir düğüm, yönlendirme kararları koşullu kenarlarda alınıyor.

---

## Hızlı Başlangıç

**Gereksinimler:** Docker ve Docker Compose (v2+), en az 8 GB RAM.

```bash
git clone https://github.com/songulerdemguler/truthRAG.git
cd truthrag
cp .env.example .env
docker compose up -d --build
```

İlk seferde modelleri çekin:

```bash
docker exec -it truthrag-ollama-1 ollama pull qwen3.5:2b
docker exec -it truthrag-ollama-1 ollama pull nomic-embed-text
```

Tarayıcıda **http://localhost:8501** adresini açın. Sidebar'dan PDF veya metin dosyası yükleyin, soru sorun, cevabı anlık izleyin.

API her başladığında önceki oturumun verilerini temizler -- sadece o oturumda yüklediğiniz dosyalar kullanılır.

---

## Temel Özellikler

**Semantik Parçalama** -- Dokümanları sabit karakter sayısına göre bölmek yerine (ki bu genellikle cümleleri ortadan keser), TruthRAG ardışık cümleler arasındaki anlam uzaklığını ölçüyor. Mesafe belirli bir eşiği aştığında yeni parça başlıyor. Böylece ilgili içerik bir arada kalıyor ve daha anlamlı arama birimleri oluşuyor.

**Cross-Encoder Yeniden Sıralama** -- Vektör araması hızlı ama yaklaşık. İlk hibrit arama sonrası, bir cross-encoder modeli (`ms-marco-MiniLM-L-6-v2`) her sorgu-parça çiftini birlikte okuyor ve daha doğru bir alaka puanı üretiyor. Çift başına daha yavaş ama "gerçekten alakalı" ile "yüzeysel olarak benzer"i ayırt etmede çok daha iyi.

**Sorgu Genişletme** -- Tek bir soru, kelime farklarından dolayı alakalı içeriği kaçırabilir. LLM sorunun 2 alternatif ifadesini üretiyor ve üç sorgu birden indekse gönderiliyor. Sonuçlar tekilleştiriliyor -- tekrarsız, daha geniş kapsam.

**RAGAS Değerlendirme** -- Test veri seti yükleyin (sorular + beklenen cevaplar) ve sayısal metrikler alın: faithfulness (cevap kaynaklara uyuyor mu?), answer relevancy (soruyu karşılıyor mu?), context recall (arama doğru içeriği buldu mu?). Hepsi lokal LLM ile hesaplanıyor.

**Parça ve Doküman Kullanım Oranları** -- Analitik paneli hangi doküman ve parçaların sorgularda en çok kullanıldığını takip ediyor. Hangi içerik cevaplara gerçekten katkı sağlıyor, hangisi boşta duruyor -- görmek için faydalı.

---

## API

Swagger dokümantasyonu: **http://localhost:8000/docs**

### Soru sor

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "RAG nedir?"}'
```

### Streaming (SSE)

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "RAG nedir?"}'
```

Token token akar: önce metadata, sonra tokenler, sonunda gecikme bilgisiyle `done` eventi.

### Sohbet oturumu

```bash
# Oturum oluştur
curl -X POST http://localhost:8000/session
# {"session_id": "a1b2c3d4e5f67890"}

# Oturumla soru sor (önceki konuşmayı hatırlar)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Bunu biraz aç", "session_id": "a1b2c3d4e5f67890"}'
```

### Doküman yükle

```bash
curl -X POST http://localhost:8000/ingest -F "file=@belge.pdf"
```

50 MB limit. PDF, TXT, MD kabul ediyor.

### Endpoint'ler

| Endpoint | Ne yapar |
|----------|----------|
| `POST /query` | Tam pipeline sorgusu (30/dk) |
| `POST /query/stream` | Streaming SSE sorgusu (30/dk) |
| `POST /session` | Sohbet oturumu oluştur |
| `POST /ingest` | Doküman yükle ve işle (10/dk) |
| `GET /analytics/summary?days=30` | Sorgu sayısı, ort. güven, gecikme, halüsinasyonlar |
| `GET /analytics/trend?days=30` | Zaman bazlı güven trendi |
| `GET /analytics/recent?limit=20` | Son sorgular ve metrikleri |
| `GET /analytics/chunk-hits?days=30` | En çok alınan parçalar |
| `GET /analytics/document-hits?days=30` | En çok alınan dokümanlar |
| `GET /analytics/eval-summary?days=30` | RAGAS değerlendirme ortalamaları |
| `GET /analytics/eval-recent?limit=20` | Son değerlendirme sonuçları |
| `POST /evaluate` | Toplu RAGAS değerlendirmesi (5/dk) |
| `GET /health` | Qdrant + Ollama erişim kontrolü (200 veya 503) |

Her yanıt izleme için `X-Correlation-ID` header'ı içerir.

---

## Yapılandırma

Tüm ayarlar `.env` dosyasından:

| Değişken | Varsayılan | Ne yapar |
|----------|-----------|----------|
| `LLM_MODEL` | `qwen3.5:2b` | Dil modeli (Ollama'da pull edilmiş olmalı) |
| `EMBED_MODEL` | `nomic-embed-text` | Gömüleme modeli |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama adresi |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant adresi |
| `QDRANT_COLLECTION` | `truthrag` | Vektör koleksiyon adı |
| `TOP_K` | `5` | Sorgu başına alınacak parça sayısı |
| `GRADE_THRESHOLD` | `0.5` | Bir parçayı tutmak için minimum alaka puanı |
| `MAX_RETRIES` | `2` | Halüsinasyonda tekrar üretme denemesi |
| `BM25_WEIGHT` | `0.3` | Hibrit aramada anahtar kelime ağırlığı |
| `VECTOR_WEIGHT` | `0.7` | Hibrit aramada vektör ağırlığı |
| `CHUNKING_STRATEGY` | `semantic` | `semantic` veya `fixed` |
| `SEMANTIC_CHUNK_THRESHOLD` | `95.0` | Semantik bölme için yüzdelik eşik |
| `FIXED_CHUNK_SIZE` | `500` | Parça başına karakter (sabit modda) |
| `FIXED_CHUNK_OVERLAP` | `50` | Sabit parçalar arası örtüşme |
| `RERANKER_ENABLED` | `true` | Cross-encoder yeniden sıralamayı aç |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder modeli |
| `RERANKER_TOP_K` | `10` | Yeniden sıralanacak aday sayısı |
| `QUERY_EXPANSION_ENABLED` | `true` | Çoklu sorgu genişletmeyi aç |
| `QUERY_EXPANSION_COUNT` | `2` | Üretilecek alternatif sorgu sayısı |
| `RAGAS_EVAL_ENABLED` | `true` | RAGAS değerlendirme endpoint'lerini aç |
| `MAX_CONVERSATION_TURNS` | `10` | Oturum başına geçmiş derinliği |
| `WEB_SEARCH_ENABLED` | `true` | Yerel doküman yetersizse web fallback |
| `WEB_SEARCH_MAX_PAGES` | `3` | Arama başına taranacak sayfa |
| `WEB_SEARCH_TIMEOUT` | `15` | Sayfa başına zaman aşımı (saniye) |
| `PDF_PARSER` | `pymupdf` | `pymupdf` (hızlı) veya `docling` (tablo/layout) |

Docker'sız çalıştırıyorsanız: `OLLAMA_BASE_URL=http://localhost:11434` ve `QDRANT_URL=http://localhost:6333` yapın.

---

## Teknoloji Yığını

| Bileşen | Araç | Neden |
|---------|------|-------|
| LLM | Ollama + qwen3.5:2b | Hızlı lokal çıkarım, API anahtarı gerektirmez |
| PDF okuma | PyMuPDF (varsayılan) / Docling (opsiyonel) | PyMuPDF anında açar; Docling tablo ve karmaşık layout'lar için |
| Gömüleme | nomic-embed-text | Boyutuna göre iyi kalite, hafif |
| Vektör DB | Qdrant | Hızlı kosinüs benzerliği araması |
| Anahtar kelime | rank-bm25 | Leksikal eşleşme için BM25Okapi |
| Yeniden sıralama | sentence-transformers (cross-encoder) | Çift bazlı alaka puanlamasıyla hassasiyet |
| Parçalama | LangChain SemanticChunker | Konu sınırlarından böler, keyfi noktalardan değil |
| Pipeline | LangGraph StateGraph | Koşullu yönlendirmeli çok adımlı orkestrasyon |
| Web arama | DuckDuckGo + Crawl4AI | API anahtarı gerektirmez, tam sayfa içerik çıkarma |
| Değerlendirme | RAGAS | Faithfulness, relevancy, recall metrikleri |
| API | FastAPI | REST + SSE streaming, rate limiting, CORS |
| Arayüz | Streamlit | Sohbet, analitik paneli, değerlendirme ekranı |
| Analitik | SQLite | Sorgu metrikleri, parça kullanım takibi |
| Altyapı | Docker Compose | Tek komutla her şey ayağa kalkar |

---

## Geliştirme

```bash
pip install -e ".[dev]"
pre-commit install

make lint          # ruff check
make type-check    # mypy
make test          # pytest
make check         # hepsi bir arada
make format        # otomatik düzelt + formatla
```

Lokal çalıştırmak için (Docker'sız):

```bash
# Sadece altyapıyı başlat
docker run -d -p 6333:6333 qdrant/qdrant
docker run -d -p 11434:11434 -v ollama_data:/root/.ollama ollama/ollama
ollama pull qwen3.5:2b && ollama pull nomic-embed-text

# API ve arayüzü yerel çalıştır
uvicorn src.api.main:app --reload --port 8000
streamlit run ui/app.py  # ayrı terminalde
```

---

## Proje Yapısı

```
truthrag/
├── docker-compose.yml
├── Dockerfile / Dockerfile.ui
├── pyproject.toml
├── Makefile
│
├── src/
│   ├── config.py                  # Tüm ayarlar (.env'den)
│   ├── utils.py                   # Singleton client'lar, JSON parse, zamanlayıcı
│   ├── analytics.py               # SQLite: sorgu logu, değerlendirme logu, parça kullanımı
│   ├── conversation.py            # Bellek içi oturum deposu
│   ├── ingestion/
│   │   ├── loader.py              # PyMuPDF / Docling (PDF), TextLoader (TXT/MD)
│   │   └── embedder.py            # Semantik veya sabit parçalama + Qdrant upsert
│   ├── retrieval/
│   │   ├── retriever.py           # Vektör + BM25 + RRF hibrit arama
│   │   └── reranker.py            # Cross-encoder yeniden sıralama
│   ├── agents/
│   │   ├── grader.py              # Parça alaka puanlaması
│   │   ├── generator.py           # Atıflı yanıt üretimi
│   │   ├── hallucination_checker.py
│   │   ├── query_expander.py      # Çoklu sorgu genişletme
│   │   └── web_search.py          # DuckDuckGo + Crawl4AI fallback
│   ├── evaluation/
│   │   └── ragas_eval.py          # RAGAS toplu + tekli değerlendirme
│   └── pipeline/
│       └── graph.py               # LangGraph StateGraph tanımı
│
├── ui/
│   └── app.py                     # Streamlit: sohbet, analitik, RAGAS değerlendirme
├── tests/
└── data/
    ├── ingest/                    # Dokümanlarınızı buraya yükleyin
    └── eval/                      # RAGAS değerlendirme veri setleri
```
