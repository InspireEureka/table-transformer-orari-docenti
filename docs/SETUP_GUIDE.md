# Guida alla Configurazione dell'Ambiente

Guida passo-passo per configurare l'ambiente di sviluppo per il fine-tuning
di TATR su orari scolastici italiani.

**Hardware testato:** NVIDIA RTX 3060 12GB VRAM, 32GB RAM, Ubuntu 22.04

---

## Prerequisiti di Sistema

| Componente | Versione minima | Consigliata |
|------------|----------------|-------------|
| OS         | Ubuntu 20.04   | Ubuntu 22.04 |
| Driver NVIDIA | 520.x       | 535.x o superiore |
| Python     | 3.9            | 3.10 |
| Spazio disco | 30 GB        | 60 GB (per dataset + checkpoint) |
| RAM        | 16 GB          | 32 GB |
| VRAM       | 8 GB           | 12 GB (RTX 3060) |

Verifica driver NVIDIA installati:
```bash
nvidia-smi
```
Output atteso: versione driver, utilizzo GPU, temperatura.

---

## Passo 1: Installazione CUDA Toolkit 11.8

TATR richiede CUDA >= 11.0. La versione 11.8 è compatibile con RTX 3060 (architettura Ampere).

```bash
# Aggiungi repository NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Installa CUDA Toolkit 11.8
sudo apt-get install cuda-toolkit-11-8

# Aggiungi al PATH (aggiungi anche a ~/.bashrc)
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Verifica installazione
nvcc --version
# Output atteso: Cuda compilation tools, release 11.8
```

---

## Passo 2: Creazione Ambiente Virtuale Python

```bash
# Clona questo repository
git clone https://github.com/InspireEureka/table-transformer-orari-docenti.git
cd table-transformer-orari-docenti

# Crea ambiente virtuale (escluso da .gitignore)
python3 -m venv .venv

# Attiva l'ambiente (da fare ogni volta che apri il terminale)
source .venv/bin/activate

# Aggiorna pip
pip install --upgrade pip

# Verifica versione Python
python --version
# Output atteso: Python 3.10.x
```

---

## Passo 3: Installazione Dipendenze Python

```bash
# Assicurati che l'ambiente virtuale sia attivo
source .venv/bin/activate

# Installa tutte le dipendenze (include torch con CUDA 11.8)
pip install -r requirements.txt

# Verifica GPU accessibile da PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# Output atteso: CUDA: True, GPU: NVIDIA GeForce RTX 3060
```

**Nota:** Il download di torch con supporto CUDA è ~2GB. Con connessione lenta
può richiedere 10-20 minuti.

---

## Passo 4: Clone Repository TATR (Microsoft)

Lo script di training originale di TATR è necessario per il fine-tuning.
L'API HuggingFace viene usata solo per l'inferenza (pre-annotazione e valutazione).

```bash
# Dalla directory del progetto
git clone https://github.com/microsoft/table-transformer.git

# Installa dipendenze aggiuntive di TATR
pip install -r table-transformer/requirements.txt

# IMPORTANTE: lo script di training usa import relativi
# Deve essere sempre eseguito dalla directory src/
# Il train_local.sh gestisce questo automaticamente
ls table-transformer/src/main.py  # deve esistere
```

---

## Passo 5: Download Modello Pre-addestrato

Il modello di partenza per il fine-tuning è `pubtabnet_structure_v1.1-all.pth`,
addestrato su PubTables-1M + FinTabNet.

```bash
# Crea cartella models/ (ignorata da .gitignore)
mkdir -p models/

# Opzione A: Download diretto dalla release GitHub di TATR
# Vai su: https://github.com/microsoft/table-transformer/releases
# Scarica: pubtabnet_structure_v1.1-all.pth
# Posiziona in: ./models/

# Opzione B: Download via Python (HuggingFace Hub)
# ATTENZIONE: questo scarica il formato .safetensors, NON il .pth usato da train_local.sh
# Utile SOLO per test di inferenza con transformers, non per il training
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='microsoft/table-transformer-structure-recognition-v1.1-all',
    local_dir='models/hf_model/'
)
print('Modello HuggingFace scaricato in models/hf_model/')
"

# Verifica presenza modello per training
ls -lh models/pubtabnet_structure_v1.1-all.pth
# Output atteso: -rw-r--r-- 1 user user 112M ...
```

**Differenza tra .pth e HuggingFace:**
- `.pth`: usato da `train_local.sh` per il fine-tuning
- HuggingFace: usato da `pre_annotate.py` e `eval_model.py` per l'inferenza
- Non sono intercambiabili (vedi `docs/ISSUE_TATR_REFERENCE.md`)

---

## Passo 6: Struttura Directory Dataset

Prima di avviare il training, il dataset deve essere organizzato così:

```
dataset/
├── images/              ← tutte le immagini (flat, nessuna sottocartella)
│   ├── orario_001.jpg
│   ├── orario_002.jpg
│   └── ...              (~2100 immagini totali)
├── train/               ← XML Pascal VOC training (80%, ~1680 file)
│   ├── orario_001.xml
│   ├── orario_002.xml
│   └── ...
├── val/                 ← XML Pascal VOC validazione (10%, ~210 file)
│   ├── orario_500.xml
│   └── ...
└── test/                ← XML Pascal VOC test (10%, ~210 file)
    ├── orario_800.xml
    └── ...
```

**Regola fondamentale:** ogni immagine `orario_NNN.jpg` deve avere il corrispondente
`orario_NNN.xml` nella stessa cartella (train, val o test). Il nome base deve coincidere.

Per generare gli XML automaticamente:
```bash
# Pre-annotazione automatica con TATR pre-addestrato
python pre_annotate.py \
    --input dataset_raw/ \
    --output dataset/train/ \
    --copia-immagini \
    --soglia 0.5

# Poi correggere in CVAT e dividere in val/ e test/ manualmente
```

---

## Passo 7: Verifica Setup Completo

```bash
# Verifica GPU, dipendenze e inferenza TATR
python test_tatr.py

# Output atteso:
# ╔═══════════════════════════════════╗
# ║  Verifica Ambiente - TATR         ║
# ╚═══════════════════════════════════╝
# [OK] GPU: NVIDIA GeForce RTX 3060
# VRAM: 12.00 GB totale | 11.xx GB libera
# [OK] torch 2.1.2+cu118
# [OK] transformers 4.38.2
# ...
# [OK] Modello caricato: 115M parametri
# [OK] Inferenza: X oggetti rilevati
# ✓ Ambiente pronto per il fine-tuning!
```

Se tutto è OK, procedi con:
1. Preparazione dataset: `python pre_annotate.py --help`
2. Annotazione manuale: vedi `docs/ANNOTAZIONE_GUIDE.md`
3. Training: `bash train_local.sh`

---

## Risoluzione Problemi

Per problemi comuni consulta `docs/TROUBLESHOOTING.md`.

I problemi più frequenti in questa fase:
- **CUDA out of memory**: riduci `batch_size` in `structure_config.json`
- **torch senza CUDA**: reinstalla con `--extra-index-url https://download.pytorch.org/whl/cu118`
- **import relativi TATR**: assicurati che `train_local.sh` faccia `cd` in `table-transformer/src/`
