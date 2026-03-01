# Guida all'Annotazione con CVAT

Guida completa per annotare le immagini di orari scolastici usando CVAT,
integrando la pre-annotazione automatica generata da `pre_annotate.py`.

**Obiettivo:** annotare ~2100 immagini in ~3 settimane, sfruttando la pre-annotazione
automatica per ridurre il tempo da 3-5 min a ~30 sec per immagine.

---

## Workflow Complessivo

```
1. Esegui pre_annotate.py
       ↓
2. Importa in CVAT (immagini + XML)
       ↓
3. Correggi annotazioni errate
       ↓
4. Esporta da CVAT (formato Pascal VOC 1.1)
       ↓
5. Converti struttura export → struttura TATR
       ↓
6. Dividi in train/val/test (80/10/10)
       ↓
7. Avvia training con train_local.sh
```

---

## Installazione CVAT (Docker)

CVAT è lo strumento di annotazione open-source consigliato, installabile localmente.

```bash
# Pre-requisito: Docker e Docker Compose installati
# https://docs.docker.com/get-docker/

# Clona CVAT
git clone https://github.com/opencv/cvat.git
cd cvat

# Avvia CVAT (prima esecuzione: scarica ~3GB di immagini Docker)
docker compose up -d

# Accesso web
# URL: http://localhost:8080
# Crea account admin alla prima visita

# Per fermare CVAT
# docker compose down
```

---

## Configurazione Etichette in CVAT

Le etichette devono corrispondere **ESATTAMENTE** alle classi TATR, spazi inclusi.
Un'etichetta `table_row` (underscore) non funzionerà.

### Configurazione JSON da incollare in CVAT

In CVAT: Crea Task → Labels → "Raw" → incolla questo JSON:

```json
[
  {
    "name": "table",
    "color": "#ffd700",
    "type": "rectangle",
    "attributes": []
  },
  {
    "name": "table row",
    "color": "#ff5050",
    "type": "rectangle",
    "attributes": []
  },
  {
    "name": "table column",
    "color": "#5050ff",
    "type": "rectangle",
    "attributes": []
  },
  {
    "name": "table column header",
    "color": "#ffa500",
    "type": "rectangle",
    "attributes": []
  },
  {
    "name": "table projected row header",
    "color": "#00b400",
    "type": "rectangle",
    "attributes": []
  },
  {
    "name": "table spanning cell",
    "color": "#a000a0",
    "type": "rectangle",
    "attributes": []
  }
]
```

---

## Importazione Annotazioni Pre-generate

Dopo aver eseguito `pre_annotate.py`, importa le pre-annotazioni in CVAT:

1. **Crea Task** in CVAT (Job Manager → Create Task)
2. **Carica immagini** dalla cartella `dataset/images/`
3. **Importa annotazioni** (Task → Actions → Upload annotations)
   - Formato: **Pascal VOC 1.1**
   - File: zip contenente gli XML generati da `pre_annotate.py`
   - Per creare lo zip: `zip -j annotazioni.zip dataset/train/*.xml`

**Attenzione:** il nome del file XML deve corrispondere al nome dell'immagine (stessa base).
Se l'immagine si chiama `orario_001.jpg`, l'XML deve chiamarsi `orario_001.xml`.

---

## Criteri di Annotazione per Orari Scolastici

### `table` — La tabella intera
- Annotare **tutta la tabella** come singolo rettangolo
- Include bordi, intestazioni e corpo
- Uno solo per immagine (salvo orari con tabelle multiple)

### `table row` — Singola riga
- Annotare **ogni riga** della tabella, incluse le righe di intestazione
- La bbox deve coprire **tutta la larghezza** della tabella
- Altezza: dal bordo superiore al bordo inferiore della riga
- **Non includere** bordi spessi tra righe nel bbox
- **Sovrapposizione con header:** normale e attesa in TATR

### `table column` — Singola colonna
- Annotare **ogni colonna**, incluse quelle con ore e giorni
- La bbox deve coprire **tutta l'altezza** della tabella
- Larghezza: dal bordo sinistro al bordo destro della colonna

### `table column header` — Riga intestazione colonne
- La **prima riga** con i nomi dei giorni (Lunedì, Martedì, ecc.)
- La bbox si **sovrappone** a una `table row`: è corretto in TATR
- Se ci sono due righe di intestazione (es. giorno + data), annotale entrambe

### `table projected row header` — Intestazione di riga proiettata
- Cella nella prima colonna con il **nome dell'ora** (es. "1ª ora", "8:00-9:00")
- Si sovrappone a una `table row` e a una `table column`: normale
- Comune negli orari: ogni ora ha la sua cella di intestazione

### `table spanning cell` — Cella che occupa più righe/colonne
- Celle che si estendono su **più righe o più colonne**
- Esempio tipico: "RICREAZIONE" che copre tutte le colonne di quell'ora
- Esempio: "LABORATORIO" che dura due ore (spanning su due righe)

---

## Suggerimenti per Velocizzare l'Annotazione

1. **Usa i tasti di scelta rapida CVAT:**
   - `N` → nuova forma
   - `1-6` → seleziona etichetta per numero
   - `F` → avanza immagine
   - `D` → immagine precedente
   - `Ctrl+Z` → annulla

2. **Correggi le pre-annotazioni invece di disegnarle da zero:**
   - Seleziona un bbox → trascina i bordi per ridimensionarlo
   - Doppio clic → elimina bbox errate
   - Priorità: correggi prima le `table row` e `table column` (le più critiche)

3. **Workflow efficiente per immagine:**
   - 30 sec: verifica visiva globale (l'auto-annotazione è corretta?)
   - 1 min: aggiusta bbox errati o mancanti
   - 10 sec: verifica e avanti

---

## Conversione Export CVAT → Struttura TATR

CVAT esporta in un formato zip diverso da quello atteso da TATR.

**Struttura export CVAT:**
```
taskname.zip/
  Annotations/
    orario_001.xml
    orario_002.xml
  JPEGImages/
    orario_001.jpg
```

**Struttura attesa da TATR:**
```
dataset/
  images/
    orario_001.jpg
  train/
    orario_001.xml
```

Script di conversione:
```python
#!/usr/bin/env python3
"""Converti export CVAT in struttura TATR."""
import shutil
import zipfile
from pathlib import Path

def converti_cvat_a_tatr(zip_path: str, output_dir: str, split: str = "train"):
    """
    Converti l'export CVAT Pascal VOC nella struttura TATR.

    Argomenti:
        zip_path: path allo zip esportato da CVAT
        output_dir: cartella dataset/ di destinazione
        split: 'train', 'val' o 'test'
    """
    output = Path(output_dir)
    dir_immagini = output / "images"
    dir_split = output / split

    dir_immagini.mkdir(parents=True, exist_ok=True)
    dir_split.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        for nome in z.namelist():
            percorso = Path(nome)
            if percorso.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                dest = dir_immagini / percorso.name
                with z.open(nome) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
            elif percorso.suffix.lower() == ".xml":
                dest = dir_split / percorso.name
                with z.open(nome) as src, open(dest, "wb") as dst:
                    dst.write(src.read())

    print(f"Conversione completata → {output}")

# Uso:
# converti_cvat_a_tatr("export_cvat.zip", "dataset/", split="train")
```

---

## Divisione Train / Val / Test

Con ~2100 immagini, usa la divisione 80/10/10:

```python
#!/usr/bin/env python3
"""Dividi il dataset in train/val/test mantenendo la distribuzione per fonte."""
import random
import shutil
from pathlib import Path

def dividi_dataset(dir_immagini: str, dir_xml: str, dir_output: str, seed: int = 42):
    """
    Dividi le immagini annotate in split train/val/test.

    Strategia: stratificata per prefisso del file (per non mescolare immagini
    dello stesso docente in split diversi).
    """
    random.seed(seed)

    xml_files = sorted(Path(dir_xml).glob("*.xml"))
    random.shuffle(xml_files)

    n = len(xml_files)
    n_train = int(n * 0.80)
    n_val = int(n * 0.10)

    split_map = {
        "train": xml_files[:n_train],
        "val": xml_files[n_train:n_train + n_val],
        "test": xml_files[n_train + n_val:]
    }

    output = Path(dir_output)
    immagini = Path(dir_immagini)

    for split, files in split_map.items():
        dir_split = output / split
        dir_split.mkdir(parents=True, exist_ok=True)
        copiati = 0
        for xml in files:
            shutil.copy2(xml, dir_split / xml.name)
            copiati += 1
        print(f"  {split}: {copiati} file")

# Uso:
# dividi_dataset("dataset/images/", "tutte_annotazioni/", "dataset/")
```

---

## Quality Check Annotazioni

Dopo l'annotazione, verifica la qualità con eval_model.py:

```bash
# Valutazione visuale su 5 immagini campione
for img in dataset/images/orario_{001,100,500,1000,2000}.jpg; do
    python eval_model.py --immagine "$img" --output quality_check/
done

# Poi visualizza i file in quality_check/ per verificare bbox
```

**Soglie di qualità:**
- Pre-annotazioni TATR base: soglia 0.5 → ~70% accuracy
- Dopo correzione manuale: target 95%+ accuracy
- Modello fine-tuned: soglia 0.7 su orari scolastici
