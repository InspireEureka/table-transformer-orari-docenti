#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# =============================================================================
# train_local.sh
# =============================================================================
# Avvio del fine-tuning di TATR su GPU locale (RTX 3060 12GB).
#
# Pre-requisiti:
#   1. Repository microsoft/table-transformer clonato in ./table-transformer/
#      git clone https://github.com/microsoft/table-transformer.git
#
#   2. Dataset preparato nella struttura corretta:
#      dataset/
#        images/     <- tutte le immagini (flat, nessuna sottocartella)
#        train/      <- XML Pascal VOC training (~1680 file, 80%)
#        val/        <- XML Pascal VOC validazione (~210 file, 10%)
#        test/       <- XML Pascal VOC test (~210 file, 10%)
#
#   3. Modello pre-addestrato scaricato in ./models/
#      Scarica manualmente da:
#      https://github.com/microsoft/table-transformer/releases
#      File: pubtabnet_structure_v1.1-all.pth
#
#   4. Ambiente virtuale attivato con dipendenze installate:
#      source .venv/bin/activate
#      pip install -r requirements.txt
#      pip install -r table-transformer/requirements.txt
#
# Uso:
#   bash train_local.sh                    # training completo (40 epoche)
#   bash train_local.sh --riprendi        # riprende dall'ultimo checkpoint
#   bash train_local.sh --valuta          # solo valutazione (no training)
#   bash train_local.sh --epoche 10       # training parziale (test pipeline)
#   bash train_local.sh --help            # mostra questo aiuto
#
# NOTA TECNICA - Perché il cd in table-transformer/src/ è obbligatorio:
#   Lo script TATR main.py usa import relativi:
#     from table_datasets import PDFTablesDataset
#     from models import build_model
#   Questi import funzionano SOLO se la working directory è table-transformer/src/
# =============================================================================

set -e  # Esci in caso di errore

# =============================================================================
# CONFIGURAZIONE VARIABILI
# =============================================================================

# Percorso di questo script (directory del progetto)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Percorso del repository TATR originale
TATR_SRC="${REPO_DIR}/table-transformer/src"

# Dataset e modelli
DATASET_DIR="${REPO_DIR}/dataset"
MODELS_DIR="${REPO_DIR}/models"
CHECKPOINT_DIR="${REPO_DIR}/checkpoints"
CONFIG_FILE="${REPO_DIR}/structure_config.json"

# Modello pre-addestrato (punto di partenza per il fine-tuning)
# Ottimizzato per struttura tabellare (righe, colonne, intestazioni)
MODELLO_BASE="${MODELS_DIR}/pubtabnet_structure_v1.1-all.pth"

# Log con timestamp (non ignorato da .gitignore — solo training_*.log)
LOG_FILE="${REPO_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

# Modalità di default
MODE="train"
RIPRENDI=false
EPOCHE_OVERRIDE=""

# =============================================================================
# FUNZIONI UTILITÀ
# =============================================================================

stampa_aiuto() {
    cat << EOF

UTILIZZO: bash train_local.sh [OPZIONI]

OPZIONI:
  --riprendi        Riprende il training dall'ultimo checkpoint in ${CHECKPOINT_DIR}/
  --valuta          Esegue solo la valutazione (mode=eval)
  --epoche N        Sovrascrive il numero di epoche da structure_config.json
  --help            Mostra questo messaggio

ESEMPI:
  bash train_local.sh
  bash train_local.sh --riprendi
  bash train_local.sh --epoche 10        # test pipeline con 10 epoche
  bash train_local.sh --valuta

FILE DI CONFIGURAZIONE: ${CONFIG_FILE}
  Modifica batch_size, epochs, lr in questo file per cambiare i parametri.

OUTPUT:
  Checkpoint: ${CHECKPOINT_DIR}/
  Log:        ${REPO_DIR}/training_YYYYMMDD_HHMMSS.log
  Metriche:   ${CHECKPOINT_DIR}/metrics.json

EOF
    exit 0
}

stampa_titolo() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Fine-Tuning TATR — Orari Scolastici Italiani               ║"
    echo "║  Hardware: RTX 3060 12GB VRAM | Dataset: ~2100 immagini     ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
}

controlla_gpu() {
    echo "  → Verifica GPU CUDA..."
    if ! python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA non disponibile'" 2>/dev/null; then
        echo "  [ERRORE] GPU CUDA non rilevata. Il training sarà molto lento su CPU."
        echo "  Verifica driver NVIDIA con: nvidia-smi"
        read -r -p "  Continuare su CPU? [s/N] " risposta
        if [[ ! "$risposta" =~ ^[Ss]$ ]]; then
            exit 1
        fi
        DEVICE="cpu"
    else
        # Stampa informazioni GPU
        python3 -c "
import torch
nome = torch.cuda.get_device_name(0)
libera, totale = torch.cuda.mem_get_info(0)
gb = 1024**3
print(f'  [OK] GPU: {nome}')
print(f'  VRAM: {totale/gb:.1f}GB totale | {libera/gb:.1f}GB libera')
"
        DEVICE="cuda"
    fi
}

controlla_prerequisiti() {
    echo "  → Verifica prerequisiti..."

    # Verifica TATR source
    if [[ ! -f "${TATR_SRC}/main.py" ]]; then
        echo "  [ERRORE] TATR non trovato in: ${TATR_SRC}"
        echo "  Esegui: git clone https://github.com/microsoft/table-transformer.git"
        exit 1
    fi
    echo "  [OK] TATR source: ${TATR_SRC}"

    # Verifica struttura dataset
    for subdir in "images" "train" "val"; do
        if [[ ! -d "${DATASET_DIR}/${subdir}" ]]; then
            echo "  [ERRORE] Mancante: ${DATASET_DIR}/${subdir}/"
            echo "  Assicurati di aver eseguito pre_annotate.py e preparato il dataset."
            exit 1
        fi
    done

    # Conta file
    N_IMMAGINI=$(find "${DATASET_DIR}/images" -maxdepth 1 \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
    N_TRAIN=$(find "${DATASET_DIR}/train" -maxdepth 1 -name "*.xml" 2>/dev/null | wc -l)
    N_VAL=$(find "${DATASET_DIR}/val" -maxdepth 1 -name "*.xml" 2>/dev/null | wc -l)
    echo "  [OK] Dataset: ${N_IMMAGINI} immagini | ${N_TRAIN} train XML | ${N_VAL} val XML"

    # Verifica configurazione
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo "  [ERRORE] Config non trovata: ${CONFIG_FILE}"
        exit 1
    fi
    echo "  [OK] Config: ${CONFIG_FILE}"

    # Crea cartella checkpoint
    mkdir -p "${CHECKPOINT_DIR}"
    echo "  [OK] Checkpoint dir: ${CHECKPOINT_DIR}"
}

trova_ultimo_checkpoint() {
    # Restituisce il path dell'ultimo checkpoint in ordine cronologico
    local ultimo
    ultimo=$(ls -t "${CHECKPOINT_DIR}"/*.pth 2>/dev/null | head -1)
    echo "${ultimo}"
}

# =============================================================================
# PARSING ARGOMENTI
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --riprendi)
            RIPRENDI=true
            shift
            ;;
        --valuta)
            MODE="eval"
            shift
            ;;
        --epoche)
            EPOCHE_OVERRIDE="$2"
            shift 2
            ;;
        --help|-h)
            stampa_aiuto
            ;;
        *)
            echo "[ERRORE] Argomento sconosciuto: $1"
            echo "Usa --help per vedere le opzioni disponibili."
            exit 1
            ;;
    esac
done

# =============================================================================
# INIZIO SCRIPT
# =============================================================================

stampa_titolo

echo "VERIFICA AMBIENTE"
echo "─────────────────────────────────────────"
controlla_gpu
controlla_prerequisiti
echo ""

# =============================================================================
# COSTRUZIONE COMANDO TRAINING
# =============================================================================

# Determina modello da caricare
if [[ "${RIPRENDI}" == true ]]; then
    CHECKPOINT=$(trova_ultimo_checkpoint)
    if [[ -z "${CHECKPOINT}" ]]; then
        echo "[ERRORE] Nessun checkpoint trovato in: ${CHECKPOINT_DIR}"
        echo "Avvia un training completo prima di usare --riprendi"
        exit 1
    fi
    echo "  Riprendendo da checkpoint: ${CHECKPOINT}"
    # Senza --load_weights_only: mantiene stato ottimizzatore e numero epoca
    ARGS_MODELLO="--model_load_path ${CHECKPOINT}"
else
    if [[ ! -f "${MODELLO_BASE}" ]]; then
        echo "[ERRORE] Modello base non trovato: ${MODELLO_BASE}"
        echo ""
        echo "Scarica il modello pre-addestrato:"
        echo "  1. Vai su: https://github.com/microsoft/table-transformer/releases"
        echo "  2. Scarica: pubtabnet_structure_v1.1-all.pth"
        echo "  3. Posizionalo in: ${MODELS_DIR}/"
        echo ""
        echo "  Oppure scarica con Python:"
        echo "  python3 -c \\"
        echo "    \"from huggingface_hub import hf_hub_download; \\"
        echo "     hf_hub_download('microsoft/table-transformer-structure-recognition-v1.1-all',\\"
        echo "     'pytorch_model.bin', local_dir='${MODELS_DIR}')\""
        exit 1
    fi
    echo "  Modello base: ${MODELLO_BASE}"
    # Con --load_weights_only: carica solo i pesi, ottimizzatore riparte da zero
    # Essenziale per fine-tuning: il LR scheduler riparte fresh
    ARGS_MODELLO="--model_load_path ${MODELLO_BASE} --load_weights_only"
fi

# Override epoche se specificato
if [[ -n "${EPOCHE_OVERRIDE}" ]]; then
    echo "  Epoche override: ${EPOCHE_OVERRIDE}"
    # Crea config temporaneo con epoche modificate
    CONFIG_TEMP="${REPO_DIR}/structure_config_temp.json"
    python3 -c "
import json
with open('${CONFIG_FILE}') as f:
    cfg = json.load(f)
cfg['epochs'] = ${EPOCHE_OVERRIDE}
with open('${CONFIG_TEMP}', 'w') as f:
    json.dump(cfg, f, indent=2)
print(f'  Config temporaneo creato: ${CONFIG_TEMP}')
"
    CONFIG_USATO="${CONFIG_TEMP}"
else
    CONFIG_USATO="${CONFIG_FILE}"
fi

# =============================================================================
# AVVIO TRAINING
# =============================================================================

echo ""
echo "AVVIO TRAINING"
echo "─────────────────────────────────────────"
echo "  Modalità  : ${MODE}"
echo "  Config    : ${CONFIG_USATO}"
echo "  Dataset   : ${DATASET_DIR}"
echo "  Output    : ${CHECKPOINT_DIR}"
echo "  Log       : ${LOG_FILE}"
echo "  Device    : ${DEVICE:-cuda}"
echo ""
echo "  Stima durata (RTX 3060, ~1680 img train, batch=6):"
echo "    10 epoche ≈ 2 ore | 40 epoche ≈ 8 ore"
echo ""
echo "  [Avvio in 3 secondi... Ctrl+C per annullare]"
sleep 3

# OBBLIGATORIO: cd in src/ prima di eseguire main.py
# (import relativi in TATR non funzionano altrimenti)
cd "${TATR_SRC}"

# Esecuzione training con output su console E file di log
python3 main.py \
    --data_type structure \
    --config_file "${CONFIG_USATO}" \
    --data_root_dir "${DATASET_DIR}" \
    ${ARGS_MODELLO} \
    --model_save_dir "${CHECKPOINT_DIR}" \
    --metrics_save_filepath "${CHECKPOINT_DIR}/metrics.json" \
    --mode "${MODE}" \
    --device "${DEVICE:-cuda}" \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

# =============================================================================
# POST-TRAINING
# =============================================================================

cd "${REPO_DIR}"

# Pulisci config temporaneo se creato
if [[ -n "${EPOCHE_OVERRIDE}" && -f "${CONFIG_TEMP}" ]]; then
    rm -f "${CONFIG_TEMP}"
fi

echo ""
echo "─────────────────────────────────────────"

if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "[ERRORE] Training terminato con codice: ${EXIT_CODE}"
    echo "  Controlla il log: ${LOG_FILE}"
    echo "  Consulta docs/TROUBLESHOOTING.md per i problemi comuni."
    exit ${EXIT_CODE}
fi

echo "[OK] Training completato con successo!"
echo ""

# Mostra miglior checkpoint
if [[ -f "${CHECKPOINT_DIR}/metrics.json" ]]; then
    echo "  Riepilogo metriche finali:"
    python3 -c "
import json
with open('${CHECKPOINT_DIR}/metrics.json') as f:
    metrics = json.load(f)
# Mostra ultime metriche disponibili
if isinstance(metrics, list) and metrics:
    ultima = metrics[-1]
    print(f'    Ultima epoca: {ultima.get(\"epoch\", \"N/A\")}')
    for k, v in ultima.items():
        if k != 'epoch' and isinstance(v, (int, float)):
            print(f'    {k}: {v:.4f}')
elif isinstance(metrics, dict):
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f'    {k}: {v:.4f}')
" 2>/dev/null || echo "  (impossibile leggere metrics.json)"
fi

# Mostra checkpoint più recente
ULTIMO_CHECKPOINT=$(trova_ultimo_checkpoint)
if [[ -n "${ULTIMO_CHECKPOINT}" ]]; then
    echo ""
    echo "  Ultimo checkpoint: ${ULTIMO_CHECKPOINT}"
    echo ""
    echo "  Per valutare il modello:"
    echo "    python3 eval_model.py --immagine foto/test.jpg"
    echo ""
    echo "  Per valutazione mAP completa:"
    echo "    bash train_local.sh --valuta"
fi

echo ""
