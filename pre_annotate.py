#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pre_annotate.py
===============
Generazione automatica di annotazioni Pascal VOC XML usando TATR pre-addestrato.

Questo script usa il modello Table Transformer pre-addestrato per analizzare
automaticamente le immagini di orari scolastici e generare annotazioni XML
nel formato Pascal VOC richiesto per il fine-tuning di TATR.

OBIETTIVO: Ridurre il tempo di annotazione manuale da 3-5 minuti a ~30 secondi
per immagine, generando una prima bozza che l'annotatore corregge in CVAT.

STRUTTURA OUTPUT ATTESA DA TATR:
    dataset/
        images/     ← immagini copiate qui (con --copia-immagini)
        train/      ← XML Pascal VOC generati qui
        val/
        test/

NOTA IMPORTANTE - Incompatibilità HuggingFace vs TATR nativo:
    Questo script usa ESCLUSIVAMENTE l'API HuggingFace (transformers) per
    l'inferenza. I checkpoint .pth del repository TATR originale NON sono
    direttamente compatibili con TableTransformerForObjectDetection a causa
    del flag 'normalize_before' diverso nell'encoder transformer.
    Vedi docs/ISSUE_TATR_REFERENCE.md per dettagli.

Uso:
    # Pre-annotazione base
    python pre_annotate.py --input foto/ --output dataset/train/

    # Con copia immagini e soglia personalizzata
    python pre_annotate.py --input foto/ --output dataset/train/ \\
        --soglia 0.6 --copia-immagini --device cuda

    # Test rapido su prime 10 immagini
    python pre_annotate.py --input foto/ --output dataset/train/ --max 10 --debug

    # Usando un modello fine-tuned locale per ri-annotare
    python pre_annotate.py --input nuove_foto/ --output dataset/val/ \\
        --modello-hf microsoft/table-transformer-structure-recognition-v1.1-all

Esempio di utilizzo tipico per il progetto:
    # Settimana 1: pre-annotazione di tutte le 2100 immagini (~17h su RTX 3060)
    python pre_annotate.py \\
        --input dataset_raw/ \\
        --output dataset/train/ \\
        --copia-immagini \\
        --soglia 0.5
    # Poi importare in CVAT per correzione manuale
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET


# ─────────────────────────────────────────────
# COSTANTI
# ─────────────────────────────────────────────

# Modello HuggingFace per structure recognition
# v1.1-all è addestrato su PubTables-1M + FinTabNet — il migliore punto di partenza
MODELLO_DEFAULT = "microsoft/table-transformer-structure-recognition-v1.1-all"

# Classi riconosciute da TATR structure recognition
# ATTENZIONE: gli spazi sono obbligatori — "table row" ≠ "table_row"
CLASSI_STRUTTURA = [
    "table",
    "table row",
    "table column",
    "table column header",
    "table projected row header",
    "table spanning cell",
    "no object",  # Questa classe viene filtrata prima di scrivere l'XML
]

# Estensioni immagine supportate
ESTENSIONI_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ─────────────────────────────────────────────
# 1. CARICAMENTO MODELLO
# ─────────────────────────────────────────────

def carica_modello(device: str, modello_hf: str = MODELLO_DEFAULT):
    """
    Carica il modello TATR structure recognition da HuggingFace Hub.

    Argomenti:
        device: 'cuda' o 'cpu'
        modello_hf: ID del modello HuggingFace o path locale

    Restituisce:
        (model, processor) pronti per l'inferenza
    """
    from transformers import TableTransformerForObjectDetection, DetrImageProcessor

    print(f"  Modello  : {modello_hf}")
    print(f"  Device   : {device}")
    print("  Caricamento... (prima volta: ~350MB da HuggingFace)")

    processor = DetrImageProcessor.from_pretrained(modello_hf)
    model = TableTransformerForObjectDetection.from_pretrained(modello_hf)
    model = model.to(device)
    model.eval()

    print(f"  [OK] Modello pronto — {sum(p.numel() for p in model.parameters()):,} parametri")
    return model, processor


# ─────────────────────────────────────────────
# 2. PREPROCESSING IMMAGINE
# ─────────────────────────────────────────────

def preprocessing_immagine(percorso_img: Path, processor):
    """
    Apre e prepara un'immagine per l'inferenza TATR.

    Gestisce automaticamente immagini RGBA, in scala di grigi e altri formati.

    Argomenti:
        percorso_img: path all'immagine
        processor: DetrImageProcessor

    Restituisce:
        (inputs, immagine_pil, (larghezza, altezza))
    """
    from PIL import Image

    immagine = Image.open(percorso_img)

    # Converti sempre in RGB (TATR è addestrato su immagini RGB)
    if immagine.mode != "RGB":
        immagine = immagine.convert("RGB")

    larghezza, altezza = immagine.size
    inputs = processor(images=immagine, return_tensors="pt")

    return inputs, immagine, (larghezza, altezza)


# ─────────────────────────────────────────────
# 3. INFERENZA
# ─────────────────────────────────────────────

def inferenza_immagine(
    model,
    inputs: dict,
    processor,
    soglia: float,
    dim_originali: tuple,
    device: str
) -> list:
    """
    Esegue l'inferenza su un'immagine e restituisce le annotazioni filtrate.

    Argomenti:
        model: modello TATR
        inputs: dizionario output di DetrImageProcessor
        processor: DetrImageProcessor
        soglia: confidenza minima per includere una rilevazione
        dim_originali: (larghezza, altezza) dell'immagine originale
        device: 'cuda' o 'cpu'

    Restituisce:
        Lista di dizionari {"label": str, "score": float, "bbox": [xmin,ymin,xmax,ymax]}
        La classe "no object" è già filtrata.
    """
    import torch

    inputs_device = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs_device)

    # Post-processing: converte logit in bbox nelle coordinate originali
    larghezza, altezza = dim_originali
    risultati = processor.post_process_object_detection(
        outputs,
        threshold=soglia,
        target_sizes=[(altezza, larghezza)]  # target_sizes vuole (H, W)
    )[0]

    id2label = model.config.id2label
    annotazioni = []

    for lbl, score, box in zip(
        risultati["labels"].cpu().tolist(),
        risultati["scores"].cpu().tolist(),
        risultati["boxes"].cpu().tolist()
    ):
        nome_classe = id2label.get(lbl, f"classe_{lbl}")

        # Filtra "no object" — gestita internamente da TATR, non va negli XML
        if nome_classe == "no object":
            continue

        # Clip delle coordinate ai bordi dell'immagine
        xmin = max(0, int(box[0]))
        ymin = max(0, int(box[1]))
        xmax = min(larghezza, int(box[2]))
        ymax = min(altezza, int(box[3]))

        # Scarta bbox degeneri (larghezza o altezza zero)
        if xmax <= xmin or ymax <= ymin:
            continue

        annotazioni.append({
            "label": nome_classe,
            "score": round(score, 4),
            "bbox": [xmin, ymin, xmax, ymax]
        })

    return annotazioni


# ─────────────────────────────────────────────
# 4. GENERAZIONE XML PASCAL VOC
# ─────────────────────────────────────────────

def crea_xml_pascal_voc(
    nome_file: str,
    larghezza: int,
    altezza: int,
    annotazioni: list
) -> str:
    """
    Genera il contenuto XML nel formato Pascal VOC richiesto da TATR.

    Struttura generata:
        <annotation>
          <folder>images</folder>
          <filename>orario_001.jpg</filename>
          <size><width>W</width><height>H</height><depth>3</depth></size>
          <object>
            <name>table row</name>
            <difficult>0</difficult>
            <bndbox><xmin>x</xmin><ymin>y</ymin><xmax>x</xmax><ymax>y</ymax></bndbox>
          </object>
          ...
        </annotation>

    Argomenti:
        nome_file: nome del file immagine (es. "orario_001.jpg")
        larghezza: larghezza immagine in pixel
        altezza: altezza immagine in pixel
        annotazioni: lista di dict {"label", "score", "bbox"}

    Restituisce:
        Stringa XML formattata con indentazione.
    """
    radice = ET.Element("annotation")

    ET.SubElement(radice, "folder").text = "images"
    ET.SubElement(radice, "filename").text = nome_file

    # Dimensioni immagine
    size_elem = ET.SubElement(radice, "size")
    ET.SubElement(size_elem, "width").text = str(larghezza)
    ET.SubElement(size_elem, "height").text = str(altezza)
    ET.SubElement(size_elem, "depth").text = "3"

    # Un elemento <object> per ogni annotazione
    for ann in annotazioni:
        obj = ET.SubElement(radice, "object")
        ET.SubElement(obj, "name").text = ann["label"]
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        xmin, ymin, xmax, ymax = ann["bbox"]
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    # Indentazione XML (Python >= 3.9)
    import sys
    if sys.version_info >= (3, 9):
        ET.indent(radice, space="  ")

    albero = ET.ElementTree(radice)
    from io import StringIO
    buffer = StringIO()
    albero.write(buffer, encoding="unicode", xml_declaration=True)
    return buffer.getvalue()


# ─────────────────────────────────────────────
# 5. PROCESSO CARTELLA
# ─────────────────────────────────────────────

def processa_cartella(args) -> dict:
    """
    Processa tutte le immagini nella cartella di input e genera gli XML.

    Argomenti:
        args: namespace argparse con tutti i parametri

    Restituisce:
        Dizionario con statistiche di annotazione.
    """
    import torch
    from tqdm import tqdm

    # Inizializzazione percorsi
    dir_input = Path(args.input)
    dir_output = Path(args.output)
    dir_output.mkdir(parents=True, exist_ok=True)

    dir_immagini = None
    if args.copia_immagini:
        # Le immagini vanno nella cartella images/ allo stesso livello di train/val/test/
        dir_immagini = dir_output.parent / "images"
        dir_immagini.mkdir(parents=True, exist_ok=True)
        print(f"  Immagini copiate in: {dir_immagini}")

    if args.debug:
        dir_debug = dir_output / "debug"
        dir_debug.mkdir(parents=True, exist_ok=True)
        print(f"  Output debug in: {dir_debug}")

    # Raccolta file immagine
    file_immagini = []
    for estensione in ESTENSIONI_IMG:
        file_immagini.extend(dir_input.rglob(f"*{estensione}"))
        file_immagini.extend(dir_input.rglob(f"*{estensione.upper()}"))
    file_immagini = sorted(set(file_immagini))

    if not file_immagini:
        print(f"  [ERRORE] Nessuna immagine trovata in: {dir_input}")
        return {}

    # Limita numero immagini se richiesto (utile per test)
    if args.max:
        file_immagini = file_immagini[:args.max]
        print(f"  [INFO] Limitato a prime {args.max} immagini (--max)")

    print(f"  Immagini trovate: {len(file_immagini)}")
    print(f"  Soglia confidenza: {args.soglia}")

    # Determina device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Carica modello
    print()
    print("Caricamento modello TATR...")
    model, processor = carica_modello(device, args.modello_hf)

    # Statistiche
    statistiche = {
        "timestamp": datetime.now().isoformat(),
        "n_immagini": len(file_immagini),
        "n_elaborate": 0,
        "n_errori": 0,
        "n_senza_oggetti": 0,
        "conteggio_per_classe": {},
        "score_totale_per_classe": {},
        "immagini_senza_oggetti": []
    }

    # Elaborazione con barra di avanzamento
    print()
    for percorso_img in tqdm(file_immagini, desc="Generazione annotazioni", unit="img"):
        try:
            # Preprocessing
            inputs, immagine_pil, (larghezza, altezza) = preprocessing_immagine(
                percorso_img, processor
            )

            # Inferenza
            annotazioni = inferenza_immagine(
                model, inputs, processor, args.soglia, (larghezza, altezza), device
            )

            # Copia immagine nella struttura dataset se richiesto
            if args.copia_immagini and dir_immagini:
                dest_img = dir_immagini / percorso_img.name
                if not dest_img.exists():
                    shutil.copy2(percorso_img, dest_img)

            # Genera e salva XML
            xml_content = crea_xml_pascal_voc(
                nome_file=percorso_img.name,
                larghezza=larghezza,
                altezza=altezza,
                annotazioni=annotazioni
            )

            percorso_xml = dir_output / (percorso_img.stem + ".xml")
            percorso_xml.write_text(xml_content, encoding="utf-8")

            # Aggiorna statistiche
            if not annotazioni:
                statistiche["n_senza_oggetti"] += 1
                statistiche["immagini_senza_oggetti"].append(percorso_img.name)
            else:
                for ann in annotazioni:
                    classe = ann["label"]
                    statistiche["conteggio_per_classe"][classe] = (
                        statistiche["conteggio_per_classe"].get(classe, 0) + 1
                    )
                    statistiche["score_totale_per_classe"][classe] = (
                        statistiche["score_totale_per_classe"].get(classe, 0.0) + ann["score"]
                    )

            # Output visuale debug
            if args.debug and annotazioni:
                _salva_debug(immagine_pil, annotazioni, dir_debug / percorso_img.name, model)

            statistiche["n_elaborate"] += 1

        except Exception as errore:
            tqdm.write(f"  [ERRORE] {percorso_img.name}: {errore}")
            statistiche["n_errori"] += 1

    return statistiche


def _salva_debug(immagine_pil, annotazioni, percorso_output, model):
    """
    Salva un'immagine con i bounding box annotati per verifica visuale.

    Argomenti:
        immagine_pil: immagine PIL originale
        annotazioni: lista di annotazioni generate
        percorso_output: dove salvare l'immagine annotata
        model: modello (per recuperare la palette colori)
    """
    from PIL import ImageDraw

    colori = {
        "table":                       (255, 215, 0),
        "table row":                   (255, 80, 80),
        "table column":                (80, 80, 255),
        "table column header":         (255, 165, 0),
        "table projected row header":  (0, 180, 0),
        "table spanning cell":         (160, 0, 160),
    }

    img_debug = immagine_pil.copy()
    draw = ImageDraw.Draw(img_debug)

    for ann in annotazioni:
        colore = colori.get(ann["label"], (128, 128, 128))
        xmin, ymin, xmax, ymax = ann["bbox"]
        draw.rectangle([xmin, ymin, xmax, ymax], outline=colore, width=3)
        etichetta = f"{ann['label']} {ann['score']:.2f}"
        draw.text((xmin + 3, ymin + 3), etichetta, fill=colore)

    img_debug.save(str(percorso_output))


# ─────────────────────────────────────────────
# 6. GENERAZIONE REPORT
# ─────────────────────────────────────────────

def genera_report(statistiche: dict, dir_output: Path):
    """
    Salva un report JSON con le statistiche di pre-annotazione e stampa il riassunto.

    Argomenti:
        statistiche: dizionario con dati raccolti durante l'elaborazione
        dir_output: cartella dove salvare il report
    """
    # Calcola score medio per classe
    score_medio = {}
    for classe, tot in statistiche.get("score_totale_per_classe", {}).items():
        n = statistiche["conteggio_per_classe"].get(classe, 1)
        score_medio[classe] = round(tot / n, 4)

    report = {
        **statistiche,
        "score_medio_per_classe": score_medio,
        "tasso_successo": (
            statistiche["n_elaborate"] / statistiche["n_immagini"]
            if statistiche["n_immagini"] > 0 else 0
        )
    }

    # Salva JSON
    percorso_report = dir_output / "pre_annotazione_report.json"
    percorso_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Stampa riassunto a console
    print()
    print("=" * 60)
    print("REPORT PRE-ANNOTAZIONE")
    print("=" * 60)
    print(f"  Immagini elaborate : {statistiche['n_elaborate']} / {statistiche['n_immagini']}")
    print(f"  Errori             : {statistiche['n_errori']}")
    print(f"  Senza oggetti      : {statistiche['n_senza_oggetti']}")
    print()
    print("  Oggetti rilevati per classe:")
    for classe, cnt in sorted(statistiche.get("conteggio_per_classe", {}).items()):
        score = score_medio.get(classe, 0)
        print(f"    {classe:<40} {cnt:>5} oggetti  (score medio: {score:.3f})")

    if statistiche.get("immagini_senza_oggetti"):
        print()
        print("  [AVVISO] Immagini senza rilevazioni (verifica manuale consigliata):")
        for nome in statistiche["immagini_senza_oggetti"][:10]:  # Mostra max 10
            print(f"    - {nome}")
        if len(statistiche["immagini_senza_oggetti"]) > 10:
            print(f"    ... e altre {len(statistiche['immagini_senza_oggetti']) - 10}")

    print()
    print(f"  Report salvato in: {percorso_report}")
    print()
    print("  Passi successivi:")
    print("  1. Importa immagini + XML in CVAT (formato: Pascal VOC 1.1)")
    print("  2. Correggi le annotazioni errate")
    print("  3. Esporta e converti al formato TATR (vedi docs/ANNOTAZIONE_GUIDE.md)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    """Punto di ingresso principale dello script di pre-annotazione."""
    parser = argparse.ArgumentParser(
        description="Pre-annotazione automatica di orari scolastici con TATR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Pre-annotazione base (80% dataset → train)
  python pre_annotate.py --input foto/ --output dataset/train/ --copia-immagini

  # Con soglia alta (meno ma più affidabili)
  python pre_annotate.py --input foto/ --output dataset/train/ --soglia 0.7

  # Test rapido su 10 immagini con output debug
  python pre_annotate.py --input foto/ --output test_out/ --max 10 --debug

  # Su CPU (senza GPU)
  python pre_annotate.py --input foto/ --output dataset/train/ --device cpu
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Cartella contenente le immagini da annotare"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Cartella dove salvare i file XML generati (es. dataset/train/)"
    )
    parser.add_argument(
        "--soglia", "-s",
        type=float,
        default=0.5,
        help="Soglia di confidenza minima (default: 0.5). Valori più alti = meno ma più affidabili"
    )
    parser.add_argument(
        "--modello-hf",
        type=str,
        default=MODELLO_DEFAULT,
        dest="modello_hf",
        help=f"ID modello HuggingFace (default: {MODELLO_DEFAULT})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device di calcolo (default: auto — usa CUDA se disponibile)"
    )
    parser.add_argument(
        "--copia-immagini",
        action="store_true",
        dest="copia_immagini",
        help="Copia le immagini in dataset/images/ (struttura TATR)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Salva immagini con bbox in output/debug/ per verifica visuale"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Limita elaborazione a N immagini (utile per test rapidi)"
    )

    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Pre-Annotazione Automatica - Table Transformer (TATR)   ║")
    print("║  Formato output: Pascal VOC XML per fine-tuning TATR     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Verifica directory input
    dir_input = Path(args.input)
    if not dir_input.exists():
        print(f"[ERRORE] Directory input non trovata: {dir_input}")
        sys.exit(1)

    print(f"  Input  : {dir_input.resolve()}")
    print(f"  Output : {Path(args.output).resolve()}")

    # Elaborazione
    statistiche = processa_cartella(args)

    if not statistiche:
        print("[ERRORE] Nessuna immagine elaborata.")
        sys.exit(1)

    # Report finale
    genera_report(statistiche, Path(args.output))


if __name__ == "__main__":
    main()
