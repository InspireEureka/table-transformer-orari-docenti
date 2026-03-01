# Risoluzione Problemi

Soluzioni ai problemi più comuni riscontrati durante il setup e il fine-tuning
di TATR su orari scolastici con RTX 3060.

---

## 1. CUDA out of memory durante il training

**Sintomo:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
(GPU 0; 11.91 GiB total capacity; 10.50 GiB already allocated)
```

**Cause:**
- `batch_size` troppo alto per 12GB VRAM con immagini ad alta risoluzione
- Immagini molto grandi (>2000px) che aumentano il consumo di memoria
- Altri processi che usano la GPU in background

**Soluzioni:**

```bash
# Soluzione 1: Riduci batch_size in structure_config.json
# Da batch_size: 6 a batch_size: 4
# Poi riavvia: bash train_local.sh

# Soluzione 2: Riduci la dimensione massima delle immagini
# In structure_config.json: "train_max_size": 800  (da 1000)

# Soluzione 3: Verifica che nessun altro processo usi la GPU
nvidia-smi
# Cerca processi con alto utilizzo VRAM nella lista

# Soluzione 4: Svuota la cache GPU
python3 -c "import torch; torch.cuda.empty_cache(); print('Cache GPU svuotata')"
```

---

## 2. ModuleNotFoundError: No module named 'table_datasets'

**Sintomo:**
```
ModuleNotFoundError: No module named 'table_datasets'
# oppure
ModuleNotFoundError: No module named 'models'
```

**Causa:**
`train_local.sh` o `python main.py` eseguiti dalla directory sbagliata.
TATR usa import relativi che funzionano SOLO dalla directory `src/`.

**Soluzione:**
```bash
# Il comando python main.py DEVE essere eseguito da table-transformer/src/
cd table-transformer/src/
python main.py ...

# train_local.sh gestisce questo automaticamente con:
# cd "${TATR_SRC}"  (dove TATR_SRC punta a table-transformer/src/)
# Non eseguire main.py direttamente: usa sempre bash train_local.sh
```

---

## 3. Dataset vuoto o XML non trovati

**Sintomo:**
```
ERROR: No samples found
# oppure
Dataset has 0 training samples
```

**Causa:**
La struttura delle directory del dataset non corrisponde a quella attesa da TATR.

**Struttura corretta da verificare:**
```bash
ls dataset/
# Output atteso: images/  train/  val/  test/

# Verifica che i nomi base coincidano
ls dataset/images/ | head -3
# orario_001.jpg, orario_002.jpg, ...

ls dataset/train/ | head -3
# orario_001.xml, orario_002.xml, ...
# I nomi SENZA estensione devono essere identici!

# Conta file per verificare corrispondenza
echo "Immagini: $(ls dataset/images/*.jpg 2>/dev/null | wc -l)"
echo "XML train: $(ls dataset/train/*.xml 2>/dev/null | wc -l)"
```

**Soluzione:**
```bash
# Rinomina i file per allineare i nomi se necessario
# Oppure usa pre_annotate.py con --copia-immagini che gestisce automaticamente
python pre_annotate.py --input foto/ --output dataset/train/ --copia-immagini
```

---

## 4. Classe "no object" genera warning negli XML

**Sintomo:**
```
Warning: Unknown label 'no object' in annotation file
# oppure comportamento strano durante il training
```

**Causa:**
La classe `"no object"` è una classe interna di TATR (indice di "sfondo") che
non deve apparire nei file XML di annotazione.

**Soluzione:**
`pre_annotate.py` filtra già questa classe automaticamente. Se la trovi nei tuoi
XML, elimina manualmente quelle righe o rigenera le annotazioni:

```bash
# Cerca XML con "no object"
grep -rl "no object" dataset/train/ dataset/val/ dataset/test/

# Rimuovi quelle righe con sed
find dataset/ -name "*.xml" -exec sed -i '/<name>no object<\/name>/,/<\/object>/d' {} \;
```

---

## 5. torch.cuda.is_available() restituisce False

**Sintomo:**
```python
import torch
torch.cuda.is_available()  # → False
```

**Cause possibili:**
1. PyTorch installato senza supporto CUDA (versione CPU-only)
2. Driver NVIDIA non aggiornati
3. CUDA toolkit non compatibile con i driver

**Soluzioni:**

```bash
# Verifica versione CUDA supportata dai driver
nvidia-smi | grep "CUDA Version"
# Esempio: CUDA Version: 12.2 — supporta anche CUDA 11.8 (backward compatible)

# Verifica che PyTorch sia installato con CUDA
python3 -c "import torch; print(torch.version.cuda)"
# Se stampa None: PyTorch è installato senza CUDA

# Reinstalla PyTorch con CUDA 11.8
pip uninstall torch torchvision torchaudio -y
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Verifica
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, versione: {torch.version.cuda}')"
```

---

## 6. Incompatibilità tra checkpoint .pth TATR e HuggingFace

**Sintomo:**
```python
# Tentativo di caricare .pth con HuggingFace
model = TableTransformerForObjectDetection.from_pretrained("./models/modello.pth")
# RuntimeError: Error(s) in loading state_dict
```

**Causa:**
I checkpoint `.pth` del repository TATR originale e i modelli HuggingFace hanno
architetture leggermente diverse. La differenza è nel flag `normalize_before`
nell'encoder transformer (issue #127 di TATR).

**Regola da seguire:**
```
pre_annotate.py    → usa SOLO HuggingFace (from_pretrained con ID HF)
eval_model.py      → usa SOLO HuggingFace
extract_schedule.py → usa SOLO HuggingFace

train_local.sh     → usa SOLO .pth con script TATR originale
```

**NON fare:**
```python
# SBAGLIATO: mescolare i due sistemi
model = TableTransformerForObjectDetection.from_pretrained("models/checkpoint.pth")
```

**Fare invece:**
```python
# CORRETTO: usa HuggingFace per inferenza
model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition-v1.1-all"
)

# Per training: usa sempre bash train_local.sh che chiama main.py di TATR
```

---

## 7. Training molto lento (< 1 batch/secondo)

**Sintomo:**
Progresso molto lento durante il training, utilizzo GPU < 50% secondo `nvidia-smi`.

**Cause:**
- `num_workers` insufficienti nel config
- Dataset su HDD lento invece di SSD
- Immagini troppo grandi che richiedono molto preprocessing

**Soluzioni:**

```bash
# Soluzione 1: Aumenta num_workers in structure_config.json
# "num_workers": 4  (valore consigliato con 32GB RAM)
# Attenzione: non superare il numero di core fisici / 2

# Soluzione 2: Verifica che il dataset sia su SSD
df -h dataset/
# Cerca la velocità di lettura del disco

# Soluzione 3: Monitora utilizzo GPU durante training
watch -n 1 nvidia-smi

# Soluzione 4: Riduci train_max_size se le immagini sono molto grandi
# "train_max_size": 800  (riduce il preprocessing)
```

---

## 8. EasyOCR scarica modelli all'avvio

**Sintomo:**
```
Downloading detection model, please wait. This may take several minutes...
```

**Spiegazione:**
EasyOCR scarica i modelli linguistici (~200MB) alla prima esecuzione.
Questo è normale comportamento, non un errore.

**Soluzione:**
```bash
# Esegui una volta con connessione internet per scaricare i modelli
python3 -c "import easyocr; reader = easyocr.Reader(['it', 'en'])"
# Poi funziona offline. Cache in: ~/.EasyOCR/ (aggiunto a .gitignore)

# Per sapere dove sono i modelli
python3 -c "import easyocr; print(easyocr.config.MODULE_PATH)"
```

---

## 9. XML generato da pre_annotate.py con encoding errato

**Sintomo:**
```
xml.etree.ElementTree.ParseError: not well-formed (invalid token)
```

**Soluzione:**
```bash
# Verifica encoding degli XML generati
file dataset/train/*.xml | head -5
# Output atteso: XML 1.0 document, UTF-8 Unicode text

# Se encoding errato, rigenera con pre_annotate.py (già corretto in v1.0+)
python pre_annotate.py --input foto/ --output dataset/train/ --soglia 0.5
```

---

## 10. pre_annotate.py molto lento su CPU

**Sintomo:**
Ogni immagine richiede 30-60 secondi invece dei 30 previsti.

**Soluzione:**
```bash
# Verifica che usi la GPU
python pre_annotate.py --input foto/ --output dataset/train/ --device cuda

# Stima tempi:
# RTX 3060 (GPU): ~30 sec/immagine → 2100 img ≈ 17.5 ore
# CPU Intel i7:   ~120 sec/immagine → 2100 img ≈ 70 ore

# Per dataset grandi, lancia overnight con:
nohup bash -c "python pre_annotate.py --input foto/ --output dataset/train/ \
    --copia-immagini" > pre_annotate.log 2>&1 &
echo "PID: $!"
```

---

## Contatti e Risorse

- **Issue TATR originale:** https://github.com/microsoft/table-transformer/issues
- **Issue HuggingFace TATR:** https://huggingface.co/microsoft/table-transformer-structure-recognition/discussions
- **Riferimenti rilevanti:** vedi `docs/ISSUE_TATR_REFERENCE.md`
