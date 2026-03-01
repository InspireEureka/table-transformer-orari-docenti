#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_model.py
=============
Valutazione qualitativa e quantitativa del modello TATR fine-tuned su
orari scolastici italiani.

Modalità disponibili:
  visuale   - Inferenza su singola immagine con visualizzazione bbox colorati
  confronto - Confronto affiancato: modello base vs fine-tuned
  errori    - Analisi degli errori su campione casuale del test set

NOTA TECNICA - Modello vs checkpoint .pth:
    Questo script usa ESCLUSIVAMENTE l'API HuggingFace per l'inferenza.
    Per valutare con le metriche GriTS ufficiali di TATR (che richiedono i
    checkpoint .pth), usa la modalità 'map' tramite lo script originale TATR:
        cd table-transformer/src/
        python main.py --mode eval --model_load_path checkpoints/model_40.pth ...

Uso:
    # Valutazione visuale su singola immagine
    python eval_model.py --immagine foto/orario_test.jpg

    # Confronto base vs fine-tuned (richiede modello HuggingFace fine-tuned caricato)
    python eval_model.py --modalita confronto --immagine foto/test.jpg \\
        --modello microsoft/table-transformer-structure-recognition-v1.1-all

    # Analisi errori su test set
    python eval_model.py --modalita errori --dataset dataset/ \\
        --modello microsoft/table-transformer-structure-recognition-v1.1-all

    # Salva output in cartella specifica
    python eval_model.py --immagine foto/test.jpg --output risultati/
"""

import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# ─────────────────────────────────────────────
# COSTANTI
# ─────────────────────────────────────────────

# Modello base pre-addestrato
MODELLO_BASE = "microsoft/table-transformer-structure-recognition-v1.1-all"

# Soglia di confidenza di default per la visualizzazione
SOGLIA_DEFAULT = 0.5

# Colori per classe (RGB) — usati in matplotlib e PIL
COLORI_CLASSE = {
    "table":                       (1.0,  0.84, 0.0),   # Giallo oro
    "table row":                   (1.0,  0.31, 0.31),  # Rosso
    "table column":                (0.31, 0.31, 1.0),   # Blu
    "table column header":         (1.0,  0.65, 0.0),   # Arancio
    "table projected row header":  (0.0,  0.70, 0.0),   # Verde
    "table spanning cell":         (0.63, 0.0,  0.63),  # Viola
}

COLORE_DEFAULT = (0.5, 0.5, 0.5)  # Grigio per classi sconosciute


# ─────────────────────────────────────────────
# 1. CARICAMENTO MODELLO
# ─────────────────────────────────────────────

def carica_modello(modello_id: str, device: str):
    """
    Carica un modello TATR da HuggingFace Hub.

    Argomenti:
        modello_id: ID HuggingFace del modello
        device: 'cuda' o 'cpu'

    Restituisce:
        (model, processor) in eval mode
    """
    from transformers import TableTransformerForObjectDetection, DetrImageProcessor

    processor = DetrImageProcessor.from_pretrained(modello_id)
    model = TableTransformerForObjectDetection.from_pretrained(modello_id)
    model = model.to(device)
    model.eval()

    return model, processor


# ─────────────────────────────────────────────
# 2. INFERENZA
# ─────────────────────────────────────────────

def esegui_inferenza(
    model,
    processor,
    percorso_img: Path,
    soglia: float,
    device: str
) -> tuple:
    """
    Esegue l'inferenza su un'immagine e restituisce i risultati strutturati.

    Argomenti:
        model: modello TATR
        processor: DetrImageProcessor
        percorso_img: path all'immagine
        soglia: soglia di confidenza minima
        device: 'cuda' o 'cpu'

    Restituisce:
        (immagine_pil, annotazioni) dove annotazioni è lista di
        {"label": str, "score": float, "bbox": [xmin,ymin,xmax,ymax]}
    """
    import torch
    from PIL import Image

    immagine = Image.open(percorso_img).convert("RGB")
    larghezza, altezza = immagine.size

    inputs = processor(images=immagine, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    risultati = processor.post_process_object_detection(
        outputs,
        threshold=soglia,
        target_sizes=[(altezza, larghezza)]
    )[0]

    id2label = model.config.id2label
    annotazioni = []

    for lbl, score, box in zip(
        risultati["labels"].cpu().tolist(),
        risultati["scores"].cpu().tolist(),
        risultati["boxes"].cpu().tolist()
    ):
        nome = id2label.get(lbl, f"classe_{lbl}")
        if nome == "no object":
            continue
        annotazioni.append({
            "label": nome,
            "score": round(float(score), 4),
            "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        })

    return immagine, annotazioni


# ─────────────────────────────────────────────
# 3. VISUALIZZAZIONE SINGOLA IMMAGINE
# ─────────────────────────────────────────────

def inferenza_con_visualizzazione(
    percorso_img: Path,
    model,
    processor,
    soglia: float,
    device: str,
    dir_output: Path
):
    """
    Esegue inferenza su una singola immagine e crea una visualizzazione a 2 pannelli.

    Pannello sinistro: immagine originale
    Pannello destro: immagine con bounding box colorati per classe

    Argomenti:
        percorso_img: path all'immagine di test
        model: modello TATR
        processor: DetrImageProcessor
        soglia: soglia di confidenza
        device: 'cuda' o 'cpu'
        dir_output: cartella dove salvare la figura

    Restituisce:
        Path al file salvato
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D
    import numpy as np

    print(f"\n  Immagine: {percorso_img}")
    print(f"  Soglia: {soglia}")

    immagine, annotazioni = esegui_inferenza(model, processor, percorso_img, soglia, device)

    # Statistiche per classe
    conteggio = {}
    for ann in annotazioni:
        conteggio[ann["label"]] = conteggio.get(ann["label"], 0) + 1

    print(f"  Oggetti rilevati: {len(annotazioni)}")
    for classe, cnt in sorted(conteggio.items()):
        print(f"    {classe:<40} {cnt:>3} oggetti")

    # Figura matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle(f"Valutazione TATR — {percorso_img.name}\n(soglia={soglia})", fontsize=13)

    # Pannello 1: originale
    ax1.imshow(immagine)
    ax1.set_title("Immagine originale", fontsize=11)
    ax1.axis("off")

    # Pannello 2: con annotazioni
    ax2.imshow(immagine)
    ax2.set_title(f"Rilevazioni TATR ({len(annotazioni)} oggetti)", fontsize=11)
    ax2.axis("off")

    for ann in annotazioni:
        xmin, ymin, xmax, ymax = ann["bbox"]
        colore = COLORI_CLASSE.get(ann["label"], COLORE_DEFAULT)
        largh_bbox = xmax - xmin
        alt_bbox = ymax - ymin

        rect = patches.Rectangle(
            (xmin, ymin), largh_bbox, alt_bbox,
            linewidth=2,
            edgecolor=colore,
            facecolor=(*colore, 0.1)  # Riempimento semi-trasparente
        )
        ax2.add_patch(rect)
        ax2.text(
            xmin + 3, ymin - 3,
            f"{ann['score']:.2f}",
            color=colore,
            fontsize=8,
            fontweight="bold"
        )

    # Legenda
    elementi_legenda = []
    for classe in sorted(conteggio.keys()):
        colore = COLORI_CLASSE.get(classe, COLORE_DEFAULT)
        elemento = Line2D([0], [0], color=colore, linewidth=3,
                         label=f"{classe} ({conteggio[classe]})")
        elementi_legenda.append(elemento)

    if elementi_legenda:
        ax2.legend(handles=elementi_legenda, loc="lower left", fontsize=9,
                  framealpha=0.8)

    plt.tight_layout()

    # Salva output
    dir_output.mkdir(parents=True, exist_ok=True)
    nome_output = percorso_img.stem + "_valutazione.jpg"
    percorso_output = dir_output / nome_output
    plt.savefig(percorso_output, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Risultato salvato: {percorso_output}")
    return percorso_output


# ─────────────────────────────────────────────
# 4. CONFRONTO MODELLI
# ─────────────────────────────────────────────

def confronto_modelli(
    percorso_img: Path,
    modello_base_id: str,
    modello_finetuned_id: str,
    soglia: float,
    device: str,
    dir_output: Path
):
    """
    Confronto affiancato tra modello base e modello fine-tuned.

    Crea una figura a 3 pannelli:
    - Pannello 1: immagine originale
    - Pannello 2: predizioni modello base
    - Pannello 3: predizioni modello fine-tuned

    Argomenti:
        percorso_img: path all'immagine di test
        modello_base_id: ID HuggingFace del modello base
        modello_finetuned_id: ID HuggingFace del modello fine-tuned
        soglia: soglia di confidenza
        device: 'cuda' o 'cpu'
        dir_output: cartella dove salvare la figura
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D

    print(f"\n  Confronto modelli su: {percorso_img}")
    print(f"  Modello base     : {modello_base_id}")
    print(f"  Modello tuned    : {modello_finetuned_id}")

    # Carica entrambi i modelli
    print("  Caricamento modello base...")
    model_base, proc_base = carica_modello(modello_base_id, device)

    print("  Caricamento modello fine-tuned...")
    model_tuned, proc_tuned = carica_modello(modello_finetuned_id, device)

    # Inferenza
    immagine, ann_base = esegui_inferenza(model_base, proc_base, percorso_img, soglia, device)
    _, ann_tuned = esegui_inferenza(model_tuned, proc_tuned, percorso_img, soglia, device)

    def conta_per_classe(annotazioni):
        conta = {}
        for a in annotazioni:
            conta[a["label"]] = conta.get(a["label"], 0) + 1
        return conta

    conta_base = conta_per_classe(ann_base)
    conta_tuned = conta_per_classe(ann_tuned)

    print(f"\n  Modello base     — {len(ann_base)} oggetti: {conta_base}")
    print(f"  Modello tuned    — {len(ann_tuned)} oggetti: {conta_tuned}")

    # Figura
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f"Confronto TATR — {percorso_img.name}", fontsize=13)

    def disegna_annotazioni(ax, img, annotazioni, titolo):
        ax.imshow(img)
        ax.set_title(titolo, fontsize=11)
        ax.axis("off")
        for ann in annotazioni:
            xmin, ymin, xmax, ymax = ann["bbox"]
            colore = COLORI_CLASSE.get(ann["label"], COLORE_DEFAULT)
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor=colore, facecolor=(*colore, 0.1)
            )
            ax.add_patch(rect)

    ax1.imshow(immagine)
    ax1.set_title("Originale", fontsize=11)
    ax1.axis("off")

    disegna_annotazioni(ax2, immagine, ann_base,
                        f"Modello base\n({len(ann_base)} oggetti)")
    disegna_annotazioni(ax3, immagine, ann_tuned,
                        f"Modello fine-tuned\n({len(ann_tuned)} oggetti)")

    plt.tight_layout()

    dir_output.mkdir(parents=True, exist_ok=True)
    nome_output = percorso_img.stem + "_confronto.jpg"
    percorso_output = dir_output / nome_output
    plt.savefig(percorso_output, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Confronto salvato: {percorso_output}")


# ─────────────────────────────────────────────
# 5. ANALISI ERRORI
# ─────────────────────────────────────────────

def carica_gt_da_xml(percorso_xml: Path) -> list:
    """
    Carica le annotazioni ground truth da un file XML Pascal VOC.

    Argomenti:
        percorso_xml: path al file XML

    Restituisce:
        Lista di classi annotate (stringhe)
    """
    try:
        albero = ET.parse(percorso_xml)
        radice = albero.getroot()
        return [obj.find("name").text for obj in radice.findall("object")]
    except Exception:
        return []


def analisi_errori(
    dir_dataset: Path,
    model,
    processor,
    soglia: float,
    device: str,
    dir_output: Path,
    n_campione: int = 20
):
    """
    Analisi degli errori su un campione casuale del test set.

    Confronta le annotazioni ground truth (XML) con le predizioni del modello
    per identificare le immagini con più falsi negativi o positivi.

    Argomenti:
        dir_dataset: cartella radice dataset (contiene images/, test/)
        model: modello TATR
        processor: DetrImageProcessor
        soglia: soglia di confidenza
        device: 'cuda' o 'cpu'
        dir_output: cartella dove salvare il report
        n_campione: numero di immagini da analizzare
    """
    dir_immagini = dir_dataset / "images"
    dir_test = dir_dataset / "test"

    if not dir_test.exists():
        print(f"  [ERRORE] Cartella test non trovata: {dir_test}")
        return

    # Raccolta file XML test con immagine corrispondente
    coppie = []
    for xml_path in sorted(dir_test.glob("*.xml")):
        for estensione in [".jpg", ".jpeg", ".png"]:
            img_path = dir_immagini / (xml_path.stem + estensione)
            if img_path.exists():
                coppie.append((img_path, xml_path))
                break

    if not coppie:
        print(f"  [ERRORE] Nessuna coppia immagine+XML trovata in {dir_test}")
        return

    # Campione casuale
    if len(coppie) > n_campione:
        campione = random.sample(coppie, n_campione)
    else:
        campione = coppie

    print(f"\n  Analisi errori su {len(campione)} immagini del test set")

    risultati = []
    falsi_negativi_totali = 0
    falsi_positivi_totali = 0

    for img_path, xml_path in campione:
        classi_gt = carica_gt_da_xml(xml_path)

        try:
            _, ann_pred = esegui_inferenza(model, processor, img_path, soglia, device)
            classi_pred = [a["label"] for a in ann_pred]

            # Calcolo recall approssimativo per classe
            conta_gt = {}
            for c in classi_gt:
                conta_gt[c] = conta_gt.get(c, 0) + 1

            conta_pred = {}
            for c in classi_pred:
                conta_pred[c] = conta_pred.get(c, 0) + 1

            # Falsi negativi = classi GT non rilevate (approssimazione)
            fn = sum(max(0, conta_gt.get(c, 0) - conta_pred.get(c, 0))
                    for c in conta_gt)
            # Falsi positivi = rilevazioni extra rispetto al GT (approssimazione)
            fp = sum(max(0, conta_pred.get(c, 0) - conta_gt.get(c, 0))
                    for c in conta_pred)

            falsi_negativi_totali += fn
            falsi_positivi_totali += fp

            risultati.append({
                "immagine": img_path.name,
                "n_gt": len(classi_gt),
                "n_pred": len(classi_pred),
                "falsi_negativi_approx": fn,
                "falsi_positivi_approx": fp
            })

        except Exception as e:
            print(f"  [ERRORE] {img_path.name}: {e}")

    # Ordina per peggiori (più FN)
    risultati.sort(key=lambda x: x["falsi_negativi_approx"], reverse=True)

    # Report
    print(f"\n  Falsi negativi totali (approx): {falsi_negativi_totali}")
    print(f"  Falsi positivi totali (approx): {falsi_positivi_totali}")
    print(f"\n  Immagini più problematiche (più falsi negativi):")
    for r in risultati[:5]:
        print(f"    {r['immagine']}: GT={r['n_gt']}, Pred={r['n_pred']}, "
              f"FN≈{r['falsi_negativi_approx']}, FP≈{r['falsi_positivi_approx']}")

    # Salva report JSON
    dir_output.mkdir(parents=True, exist_ok=True)
    percorso_report = dir_output / "analisi_errori.json"
    percorso_report.write_text(
        json.dumps({
            "n_campione": len(campione),
            "soglia": soglia,
            "falsi_negativi_totali": falsi_negativi_totali,
            "falsi_positivi_totali": falsi_positivi_totali,
            "risultati": risultati
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n  Report salvato: {percorso_report}")
    print("\n  NOTA: valori FN/FP sono approssimazioni basate sul conteggio per classe.")
    print("        Per mAP esatta usa lo script TATR originale (train_local.sh --valuta).")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    """Punto di ingresso principale dello script di valutazione."""
    parser = argparse.ArgumentParser(
        description="Valutazione del modello TATR fine-tuned su orari scolastici",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modalità:
  visuale   Visualizza inferenza su singola immagine (default)
  confronto Confronto affiancato base vs fine-tuned
  errori    Analisi errori su campione test set

Esempi:
  python eval_model.py --immagine foto/test.jpg
  python eval_model.py --modalita confronto --immagine test.jpg \\
      --modello microsoft/table-transformer-structure-recognition-v1.1-all
  python eval_model.py --modalita errori --dataset dataset/ --soglia 0.6
        """
    )

    parser.add_argument(
        "--immagine",
        type=str,
        default=None,
        help="Path all'immagine di test (richiesto per modalità 'visuale' e 'confronto')"
    )
    parser.add_argument(
        "--modello",
        type=str,
        default=MODELLO_BASE,
        help=f"ID modello HuggingFace (default: {MODELLO_BASE})"
    )
    parser.add_argument(
        "--modello-base",
        type=str,
        default=MODELLO_BASE,
        dest="modello_base",
        help="Modello base per modalità 'confronto'"
    )
    parser.add_argument(
        "--modalita",
        type=str,
        default="visuale",
        choices=["visuale", "confronto", "errori"],
        help="Modalità di valutazione (default: visuale)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Cartella dataset (richiesto per modalità 'errori')"
    )
    parser.add_argument(
        "--soglia",
        type=float,
        default=SOGLIA_DEFAULT,
        help=f"Soglia confidenza (default: {SOGLIA_DEFAULT})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_output",
        help="Cartella output per risultati (default: eval_output/)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda' o 'cpu' (default: auto)"
    )
    parser.add_argument(
        "--campione",
        type=int,
        default=20,
        help="N immagini per analisi errori (default: 20)"
    )

    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Valutazione Modello - Table Transformer (TATR)          ║")
    print(f"║  Modalità: {args.modalita:<47}║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Determina device
    if args.device:
        device = args.device
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    print(f"\n  Device: {device}")

    dir_output = Path(args.output)

    if args.modalita == "visuale":
        if not args.immagine:
            print("[ERRORE] --immagine è obbligatorio per la modalità 'visuale'")
            sys.exit(1)
        print(f"  Caricamento modello: {args.modello}")
        model, processor = carica_modello(args.modello, device)
        inferenza_con_visualizzazione(
            Path(args.immagine), model, processor, args.soglia, device, dir_output
        )

    elif args.modalita == "confronto":
        if not args.immagine:
            print("[ERRORE] --immagine è obbligatorio per la modalità 'confronto'")
            sys.exit(1)
        confronto_modelli(
            Path(args.immagine),
            args.modello_base,
            args.modello,
            args.soglia,
            device,
            dir_output
        )

    elif args.modalita == "errori":
        if not args.dataset:
            print("[ERRORE] --dataset è obbligatorio per la modalità 'errori'")
            sys.exit(1)
        print(f"  Caricamento modello: {args.modello}")
        model, processor = carica_modello(args.modello, device)
        analisi_errori(
            Path(args.dataset),
            model,
            processor,
            args.soglia,
            device,
            dir_output,
            n_campione=args.campione
        )

    print()
    print("✓ Valutazione completata.")


if __name__ == "__main__":
    main()
