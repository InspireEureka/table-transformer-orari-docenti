# Riferimenti alle Issue GitHub di TATR

Questa pagina raccoglie i riferimenti alle issue rilevanti del repository
[microsoft/table-transformer](https://github.com/microsoft/table-transformer)
e delle discussioni HuggingFace, utili per il fine-tuning su dataset personalizzati
di orari scolastici.

---

## Issue Critiche (da leggere prima di iniziare)

### Issue #127 — Incompatibilità .pth vs HuggingFace
**URL:** https://github.com/microsoft/table-transformer/issues/127
**Rilevanza:** CRITICA

Documenta l'incompatibilità tra i checkpoint `.pth` del repository originale
e i modelli caricati tramite HuggingFace Transformers. La differenza risiede
nel flag `normalize_before` nell'encoder transformer.

**Impatto su questo progetto:**
- `pre_annotate.py` e `eval_model.py`: usano **sempre** HuggingFace API
- `train_local.sh`: usa **sempre** lo script TATR originale con `.pth`
- Non mescolare i due approcci nella stessa pipeline

**Soluzione adottata in questo progetto:**
```
Inferenza (test/valutazione): HuggingFace TableTransformerForObjectDetection
Training (fine-tuning):       TATR main.py + checkpoint .pth
```

---

### Issue #169 — Fine-tuning con dataset proprietario
**URL:** https://github.com/microsoft/table-transformer/issues/169

Descrive il processo completo di fine-tuning su un dataset personalizzato,
inclusa la conversione del dataset nel formato Pascal VOC e l'uso di
`process_fintabnet.py` come riferimento.

**Estratto rilevante:**
> "The recommended approach for custom datasets is to use the Pascal VOC XML format.
> Place images in `images/` and annotations in `train/`, `val/`, `test/` with
> matching basenames."

**Conferma:** la struttura dataset usata in questo progetto è quella raccomandata.

---

## Issue Rilevanti per il Training

### Issue #143 — Parametri ottimali per dataset piccoli
**URL:** https://github.com/microsoft/table-transformer/issues/143

Discussione sui parametri di training per dataset con < 5000 immagini.
Suggerisce `lr_drop` al 50-60% delle epoche totali per dataset piccoli.

**Per questo progetto (2100 img, 40 epoche):**
- `lr_drop: 15` (37.5% delle epoche) — scelto nel `structure_config.json`
- Alternativa discussa nell'issue: `lr_drop: 20` (50% epoche)

---

### Issue #98 — Classe "table" nel structure recognition
**URL:** https://github.com/microsoft/table-transformer/issues/98

Chiarisce che la classe `"table"` nel modello di structure recognition delimita
il confine della tabella già ritagliata (non è il modello di detection).
Per immagini di orari fotografati interi, si consiglia di:
1. Usare prima il modello detection per trovare la tabella
2. Poi applicare il structure recognition sul ritaglio

**Impatto:** `extract_schedule.py` dovrebbe usare entrambi i modelli in sequenza.

---

### Issue #201 — num_workers e DataLoader su Windows
**URL:** https://github.com/microsoft/table-transformer/issues/201

Problema con `num_workers > 0` su Windows (freeze del DataLoader).
Non rilevante per Ubuntu, ma utile se si sviluppa su Windows.

**Per Ubuntu (questo progetto):** `num_workers: 4` funziona correttamente.

---

### Issue #156 — Checkpoint migliore non sempre l'ultimo
**URL:** https://github.com/microsoft/table-transformer/issues/156

Il checkpoint finale (epoca 40) non è necessariamente il migliore.
TATR salva i checkpoint ogni `checkpoint_freq` epoche (impostato a 5 in questo progetto).

**Raccomandazione:** monitora `metrics.json` per identificare l'epoca con la
migliore AP50 sul validation set, e usa quel checkpoint.

```bash
# Script rapido per trovare la migliore epoca
python3 -c "
import json
with open('checkpoints/metrics.json') as f:
    metrics = json.load(f)
if isinstance(metrics, list):
    migliore = max(metrics, key=lambda x: x.get('AP50', 0))
    print(f'Migliore epoca: {migliore.get(\"epoch\")}, AP50: {migliore.get(\"AP50\", 0):.4f}')
"
```

---

### Issue #178 — Annotare celle sovrapposte in CVAT
**URL:** https://github.com/microsoft/table-transformer/issues/178

TATR è progettato per avere bbox sovrapposti (es. `table row` + `table column header`
si sovrappongono). Questo è atteso e non va evitato durante l'annotazione.

**Regola:** non preoccuparsi delle sovrapposizioni in CVAT — TATR le gestisce
tramite Hungarian matching durante il training.

---

## Discussioni HuggingFace

### HF Discussion #1 — Fine-tuning con Label Studio
**URL:** https://huggingface.co/microsoft/table-transformer-structure-recognition/discussions/1

Alternativa a CVAT: uso di Label Studio per l'annotazione con export diretto
in Pascal VOC. Vantaggioso per chi preferisce un'interfaccia web più moderna.

**Formato export:** Pascal VOC 1.1 — compatibile con `pre_annotate.py`.

---

### HF Discussion #16 — Fine-tuning con HuggingFace Trainer
**URL:** https://huggingface.co/microsoft/table-transformer-detection/discussions/16

Approccio alternativo al training: usare `transformers.Trainer` invece dello
script TATR originale. Vantaggi: più semplice da configurare, supporta
logging su Weights & Biases.

**Svantaggio:** non include le metriche GriTS specifiche per tabelle.

**Rilevante se:** si vuole evitare la gestione dei checkpoint `.pth` e usare
solo l'ecosistema HuggingFace.

---

### HF Discussion #23 — Convertire checkpoint .pth in formato HF
**URL:** https://huggingface.co/microsoft/table-transformer-structure-recognition/discussions/23

Script Python per convertire un checkpoint fine-tuned `.pth` nel formato
HuggingFace (`.safetensors`), permettendo di usarlo con `from_pretrained()`.

**Utile per:** condividere il modello fine-tuned su HuggingFace Hub dopo
il training con lo script TATR originale.

---

## Tutorial e Notebook Esterni

### Niels Rogge — Fine-tuning DETR su dataset custom
**URL:** https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR

TATR è basato su DETR. Questo notebook ufficiale HuggingFace mostra come
fare fine-tuning di modelli DETR-based su dataset personalizzati usando
l'API `transformers`.

**Applicabile a TATR:** sì, con adattamenti per le 7 classi TATR.

---

### TATR Paper (CVPR 2022)
**URL:** https://arxiv.org/abs/2110.00061

Paper originale: "PubTables-1M: Towards Comprehensive Table Extraction From Unstructured Documents"

Spiega l'architettura DETR+ResNet18, le metriche GriTS, e il dataset di training.
Utile per capire le scelte architetturali e le aspettative di performance.

---

## Riepilogo: Quali Issue Leggere Prima di Iniziare

| Priorità | Issue | Motivo |
|----------|-------|--------|
| 🔴 CRITICA | #127 | Incompatibilità .pth/HF — evita errori di caricamento |
| 🟠 ALTA | #169 | Formato dataset corretto per fine-tuning |
| 🟡 MEDIA | #156 | Non usare sempre l'ultimo checkpoint |
| 🟡 MEDIA | #98 | Usare detection prima di structure recognition |
| 🟢 BASSA | #178 | Le sovrapposizioni bbox sono normali |
| 🟢 BASSA | #201 | Solo se si sviluppa su Windows |
