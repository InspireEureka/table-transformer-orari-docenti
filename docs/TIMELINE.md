# Timeline del Progetto — Fine-Tuning TATR per Orari Scolastici

**Durata totale:** 3 settimane
**Hardware:** RTX 3060 12GB VRAM, 32GB RAM
**Dataset:** ~2100 immagini (100 personali + 2000 da colleghi)
**Obiettivo:** modello TATR fine-tuned con mAP@50 ≥ 0.75 su orari italiani

---

## Gantt Chart — Vista Settimanale

```
Attività                              | Sett.1 | Sett.2 | Sett.3
--------------------------------------|--------|--------|-------
1. Setup ambiente e dipendenze        | ██████ |        |
2. Verifica con test_tatr.py          | ████   |        |
3. Clone TATR + test pipeline         | ████   |        |
4. Pre-annotazione automatica         |    ████|        |
5. Setup CVAT + import pre-annot.     |    ████|        |
6. Annotazione manuale CVAT           |    ████| ██████ |
7. Conversione dataset → formato TATR |        | ████   |
8. Prima run training (10 epoche)     |        |   ████ |
9. Analisi metriche intermedie        |        |    ████|
10. Training completo (40 epoche)     |        |    ████| ████
11. Valutazione (eval_model.py)       |        |        |  ████
12. Test su nuove immagini reali      |        |        |   ████
13. Integrazione extract_schedule.py  |        |        |    ███
```

---

## Settimana 1 — Infrastruttura e Pre-annotazione

### Giorno 1-2: Setup Ambiente (8-10 ore)

**Obiettivo:** ambiente funzionante, GPU verificata, dipendenze installate.

```bash
# Sequenza di comandi da eseguire in ordine
git clone https://github.com/InspireEureka/table-transformer-orari-docenti.git
cd table-transformer-orari-docenti
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python test_tatr.py

# Poi clona TATR originale
git clone https://github.com/microsoft/table-transformer.git
pip install -r table-transformer/requirements.txt
```

**Milestone:** `python test_tatr.py` completa senza errori e mostra GPU RTX 3060.

### Giorno 3-4: Pre-annotazione Automatica (2-18 ore)

**Obiettivo:** generare bozze XML per tutte le 2100 immagini.

```bash
# Prima: organizza le immagini in una cartella
mkdir -p dataset_raw/
# Copia tutte le 100 foto personali + 2000 da colleghi in dataset_raw/

# Avvia pre-annotazione (stimato: ~17h su RTX 3060)
# Lancia in background per overnight
nohup python pre_annotate.py \
    --input dataset_raw/ \
    --output dataset/train/ \
    --copia-immagini \
    --soglia 0.5 \
    > pre_annotate.log 2>&1 &

echo "PID: $! — controlla il progresso con: tail -f pre_annotate.log"
```

**Stima:** RTX 3060 → ~30 sec/immagine → 2100 img ≈ 17.5 ore totali.
**Consiglio:** avvia il venerdì sera, trova i risultati il sabato mattina.

### Giorno 5: Setup CVAT

```bash
# Installa CVAT con Docker
git clone https://github.com/opencv/cvat.git ~/cvat
cd ~/cvat
docker compose up -d
# Apri http://localhost:8080 e crea account admin

# Configura etichette TATR in CVAT (vedi docs/ANNOTAZIONE_GUIDE.md)
```

---

## Settimana 2 — Annotazione e Prime Sperimentazioni

### Giorno 1-3: Annotazione Intensiva (6+ ore/giorno)

**Obiettivo:** correggere le pre-annotazioni di ~300 immagini (test set prioritario).

**Target giornaliero:** 100 immagini/giorno × 3 giorni = 300 immagini

```
Workflow per ogni immagine:
  30 sec → verifica visiva (l'auto-annotazione è OK?)
  60 sec → correggi bbox errati o mancanti
  10 sec → avanti alla prossima

Totale: ~100 sec/immagine = ~3 ore per 100 immagini (con pause)
```

**Priorità annotazione:** annotare PRIMA le 210 immagini che diventeranno il
test set — queste vengono valutate con `eval_model.py` e devono essere accurate.

### Giorno 4: Conversione Dataset e Preparazione Training

```bash
# Export da CVAT in formato Pascal VOC 1.1
# (da interfaccia CVAT: Task → Actions → Export annotations → Pascal VOC 1.1)

# Conversione struttura CVAT → TATR
python3 - << 'EOF'
# Usa lo script di conversione in docs/ANNOTAZIONE_GUIDE.md
EOF

# Verifica struttura dataset
python3 -c "
from pathlib import Path
for split in ['train', 'val', 'test']:
    n = len(list(Path(f'dataset/{split}').glob('*.xml')))
    print(f'{split}: {n} file XML')
n_img = len(list(Path('dataset/images').glob('*.jpg')))
print(f'images: {n_img} immagini')
"
```

### Giorno 5: Prima Run Training (10 epoche — test pipeline)

```bash
# Test con 10 epoche per verificare che la pipeline funzioni
bash train_local.sh --epoche 10

# Stima: ~2 ore su RTX 3060 con 1680 immagini training, batch_size=6
# Se completa senza errori: pipeline OK
# Se fallisce: consulta docs/TROUBLESHOOTING.md
```

---

## Settimana 3 — Training Completo e Valutazione

### Giorno 1-2: Completamento Annotazione

**Obiettivo:** raggiungere ~800 immagini annotate e verificate (38% del dataset).

**Strategia:** con 800 immagini ben annotate si ottiene già un buon fine-tuning.
Le restanti 1300 pre-annotate non verificate possono essere usate come training
aggiuntivo con soglia di qualità 0.6+.

### Giorno 3-4: Training Completo (40 epoche)

```bash
# Avvia training completo
bash train_local.sh

# Stima: ~8 ore su RTX 3060
# Lancia la mattina, trova i risultati la sera

# Monitora in tempo reale (in un altro terminale)
watch -n 5 'tail -5 training_*.log'

# Monitora utilizzo GPU
nvidia-smi -l 5
```

**Checkpoint intermedi:** salvati ogni 5 epoche in `checkpoints/`.
Se la val_loss peggiora dopo l'epoca 25-30, puoi fermare il training
e usare il checkpoint migliore.

### Giorno 5: Valutazione e Integrazione

```bash
# Valutazione qualitativa su immagini di test
for img in dataset/images/orario_{001,200,500,1000,2000}.jpg; do
    python eval_model.py --immagine "$img" --output valutazione_finale/
done

# Analisi errori sistematici
python eval_model.py --modalita errori --dataset dataset/ --campione 30

# Visualizza risultati in valutazione_finale/
ls valutazione_finale/
```

---

## Metriche di Successo

| Metrica | Baseline (v1.1-all su orari) | Target dopo Fine-tuning |
|---------|------------------------------|------------------------|
| mAP@50 (struttura) | ~0.55–0.65 (stima) | ≥ 0.75 |
| AP "table row" | ~0.60 (stima) | ≥ 0.80 |
| AP "table column" | ~0.55 (stima) | ≥ 0.75 |
| AP "table col. header" | ~0.50 (stima) | ≥ 0.70 |
| Tempo inferenza (GPU) | ~0.2 sec | invariato |
| Tempo inferenza (CPU) | ~1.5 sec | invariato |

**Nota:** le metriche baseline sono stime per immagini di orari scolastici italiani.
Il modello v1.1-all è addestrato su documenti accademici (PubTables-1M) e
finanziari (FinTabNet) — orari scolastici fotografati sono un dominio molto diverso.

---

## Riepilogo Stime Temporali

| Attività | Durata stimata | Note |
|----------|---------------|------|
| Setup ambiente | 4–8 ore | Una tantum |
| Pre-annotazione 2100 img | 17–18 ore | Overnight su RTX 3060 |
| Annotazione manuale (800 img) | 15–20 ore | ~100 img/giorno |
| Conversione dataset | 1–2 ore | Scriptata |
| Training 10 epoche (test) | ~2 ore | Verifica pipeline |
| Training 40 epoche (completo) | ~8 ore | Overnight |
| Valutazione e analisi | 3–4 ore | |
| **Totale** | **~50–62 ore** | **~3 settimane a tempo parziale** |
