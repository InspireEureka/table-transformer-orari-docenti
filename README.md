# Table Transformer per Orari Scolastici Italiani

Fine-tuning del modello **Table Transformer (TATR)** di Microsoft per il
riconoscimento automatico della struttura di orari scolastici fotografati.

> **Stato del progetto:** in sviluppo attivo — Timeline 3 settimane

---

## Panoramica

Gli orari scolastici italiani sono distribuiti spesso come foto appese in aula
o scansioni di qualità variabile. Questo progetto adatta il modello TATR
(addestrato su documenti accademici) per riconoscere la struttura tabellare
tipica degli orari docenti italiani: righe con le ore, colonne con i giorni,
intestazioni, celle che si estendono su più ora (ricreazione, laboratorio).

**Pipeline completa:**
```
Foto orario → [TATR detection] → [TATR structure] → JSON strutturato
                                        ↑
                               fine-tuned su 2100 foto
                               di orari scolastici italiani
```

---

## Avvio Rapido

```bash
# 1. Clona il repository
git clone https://github.com/InspireEureka/table-transformer-orari-docenti.git
cd table-transformer-orari-docenti

# 2. Crea ambiente virtuale e installa dipendenze
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Verifica GPU e ambiente
python test_tatr.py

# 4. Pre-annota le tue immagini
python pre_annotate.py --input foto/ --output dataset/train/ --copia-immagini

# 5. Avvia il fine-tuning (dopo aver annotato e preparato il dataset)
bash train_local.sh
```

---

## Hardware

| Componente | Specifica |
|------------|-----------|
| GPU | NVIDIA RTX 3060 12GB VRAM |
| RAM | 32GB DDR4 |
| CUDA | 11.8 |
| OS | Ubuntu 22.04 |

**Requisiti minimi:** GPU con 8GB VRAM (ridurre `batch_size` a 4 nel config).

---

## Dataset

| Fonte | Immagini | Note |
|-------|----------|------|
| Foto personali | 100 | Varie condizioni di luce e prospettiva |
| Foto da colleghi | 2000 | Diverse scuole e formati orario |
| **Totale** | **2100** | |

**Divisione:** 80% training (1680) · 10% validazione (210) · 10% test (210)

**Formato annotazioni:** Pascal VOC XML — generato automaticamente da `pre_annotate.py`,
poi corretto manualmente in CVAT.

---

## Struttura del Repository

```
table-transformer-orari-docenti/
│
├── README.md                    ← questo file
├── requirements.txt             ← dipendenze Python (torch CUDA, transformers, ecc.)
├── structure_config.json        ← config fine-tuning (batch=6, epochs=40, lr=5e-5)
├── .gitignore                   ← esclude models/, dataset/, checkpoints/
│
├── test_tatr.py                 ← verifica GPU, dipendenze, inferenza TATR
├── pre_annotate.py              ← genera annotazioni XML automatiche con TATR
├── eval_model.py                ← valutazione modello (visuale, confronto, errori)
├── train_local.sh               ← avvia fine-tuning su GPU locale
├── extract_schedule.py          ← estrae dati strutturati da un orario
│
└── docs/
    ├── SETUP_GUIDE.md           ← guida setup ambiente passo-passo
    ├── ANNOTAZIONE_GUIDE.md     ← guida annotazione con CVAT
    ├── TROUBLESHOOTING.md       ← soluzione problemi comuni
    ├── ISSUE_TATR_REFERENCE.md  ← riferimenti issue GitHub TATR
    └── TIMELINE.md              ← gantt chart 3 settimane
```

**Cartelle create automaticamente (escluse da git):**
```
models/       ← modelli pre-addestrati (.pth e HuggingFace)
dataset/      ← immagini + annotazioni XML
checkpoints/  ← checkpoint del training
```

---

## Architettura TATR

**Table Transformer (TATR)** è basato su DETR (DEtection TRansformer):
- **Backbone CNN:** ResNet-18 (estrattore di feature leggero)
- **Encoder-Decoder Transformer:** attenzione globale per rilevare strutture
- **Output:** bounding box + classe per ogni elemento della tabella

### Le 7 Classi Riconosciute

| Classe | Colore | Descrizione |
|--------|--------|-------------|
| `table` | Giallo | Perimetro dell'intera tabella |
| `table row` | Rosso | Singola riga (copre tutta la larghezza) |
| `table column` | Blu | Singola colonna (copre tutta l'altezza) |
| `table column header` | Arancio | Riga con nomi giorni (Lunedì, Martedì...) |
| `table projected row header` | Verde | Cella prima colonna con ora (1ª ora, 8:00-9:00) |
| `table spanning cell` | Viola | Cella su più righe/colonne (Ricreazione, Lab) |
| `no object` | — | Classe interna, non usata nelle annotazioni |

**Nota:** le bbox si sovrappongono per design (es. `table row` + `table column header`).
È comportamento atteso in TATR, non va evitato.

---

## Pipeline Completa

### Fase 1 — Pre-annotazione Automatica

```bash
# Genera bozze XML per ~30 sec/immagine su RTX 3060
# Lancia overnight per 2100 immagini (~17 ore)
python pre_annotate.py \
    --input dataset_raw/ \
    --output dataset/train/ \
    --copia-immagini \
    --soglia 0.5
```

### Fase 2 — Annotazione Manuale in CVAT

1. Installa CVAT con Docker (vedi `docs/ANNOTAZIONE_GUIDE.md`)
2. Importa immagini + XML pre-generati (formato Pascal VOC 1.1)
3. Correggi le annotazioni errate (~30 sec/immagine invece di 3-5 min)
4. Esporta e converti nella struttura TATR

### Fase 3 — Fine-tuning

```bash
# Scarica modello pre-addestrato in models/
# (vedi docs/SETUP_GUIDE.md Passo 5)

# Avvia training completo (40 epoche, ~8 ore su RTX 3060)
bash train_local.sh

# Test rapido con 10 epoche per verificare la pipeline
bash train_local.sh --epoche 10

# Riprendi training interrotto
bash train_local.sh --riprendi
```

### Fase 4 — Valutazione

```bash
# Visualizzazione su immagine singola
python eval_model.py --immagine foto/orario_test.jpg

# Confronto base vs fine-tuned
python eval_model.py --modalita confronto --immagine foto/test.jpg

# Analisi errori su test set
python eval_model.py --modalita errori --dataset dataset/
```

---

## Configurazione Training

Il file `structure_config.json` contiene i parametri ottimizzati per RTX 3060:

```json
{
  "backbone": "resnet18",
  "batch_size": 6,
  "epochs": 40,
  "lr": 0.00005,
  "lr_drop": 15,
  "train_max_size": 1000,
  "num_workers": 4,
  "checkpoint_freq": 5
}
```

**Note sui parametri:**
- `batch_size: 6` → calibrato per 12GB VRAM con immagini max 1000px
- `lr_drop: 15` → il learning rate viene moltiplicato per `lr_gamma` all'epoca 15
- `checkpoint_freq: 5` → salva checkpoint ogni 5 epoche (8 checkpoint totali)
- `num_workers: 4` → parallelizzazione DataLoader con 32GB RAM

---

## Note Tecniche Importanti

### Incompatibilità HuggingFace vs TATR Nativo

I checkpoint `.pth` del repository TATR originale **non sono compatibili** con
`TableTransformerForObjectDetection.from_pretrained()` a causa di differenze
architetturali (flag `normalize_before`).

| Script | API usata | Motivo |
|--------|-----------|--------|
| `pre_annotate.py` | HuggingFace | Semplicità + compatibilità |
| `eval_model.py` | HuggingFace | Semplicità + compatibilità |
| `train_local.sh` | TATR nativo + `.pth` | Metriche GriTS + checkpointing |

Vedi `docs/ISSUE_TATR_REFERENCE.md` → Issue #127 per dettagli.

### Import Relativi in TATR

Lo script `train_local.sh` esegue `cd table-transformer/src/` prima di chiamare
`python main.py`. Questo è obbligatorio: TATR usa `from table_datasets import ...`
(import relativo) che funziona solo dalla directory `src/`.

---

## Guide Dettagliate

| Documento | Contenuto |
|-----------|-----------|
| [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) | Setup CUDA, Python, dipendenze passo-passo |
| [docs/ANNOTAZIONE_GUIDE.md](docs/ANNOTAZIONE_GUIDE.md) | CVAT, criteri per ogni classe, conversione dataset |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | 10 problemi comuni con soluzioni |
| [docs/ISSUE_TATR_REFERENCE.md](docs/ISSUE_TATR_REFERENCE.md) | Issue GitHub TATR rilevanti per questo progetto |
| [docs/TIMELINE.md](docs/TIMELINE.md) | Gantt chart 3 settimane con stime temporali |

---

## Issue GitHub di Riferimento (TATR)

| Issue | Rilevanza |
|-------|-----------|
| [#127 — Incompatibilità .pth/HF](https://github.com/microsoft/table-transformer/issues/127) | CRITICA |
| [#169 — Fine-tuning dataset proprietario](https://github.com/microsoft/table-transformer/issues/169) | ALTA |
| [#156 — Checkpoint migliore](https://github.com/microsoft/table-transformer/issues/156) | MEDIA |
| [#98 — Classe table in structure recognition](https://github.com/microsoft/table-transformer/issues/98) | MEDIA |

---

## Licenza

Questo progetto è rilasciato sotto licenza **Apache 2.0**, compatibile con
la licenza del repository TATR originale (microsoft/table-transformer).

Il modello base (`microsoft/table-transformer-*`) è soggetto alla licenza MIT
di Microsoft Research.
