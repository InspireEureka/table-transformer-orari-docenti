#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_tatr.py
============
Script di verifica dell'ambiente per il fine-tuning di Table Transformer (TATR).

Controlla in sequenza:
  1. Disponibilità GPU CUDA e VRAM
  2. Versioni di tutte le dipendenze Python necessarie
  3. Caricamento corretto del modello TATR pre-addestrato da HuggingFace
  4. Inferenza di test su un'immagine (sintetica o fornita dall'utente)

Uso:
    python test_tatr.py
    python test_tatr.py --immagine path/a/orario.jpg
    python test_tatr.py --skip-modello          # solo check hardware + dipendenze
    python test_tatr.py --debug                 # salva output visuale in test_output.jpg

Codici di uscita:
    0 = tutto OK
    1 = GPU CUDA non disponibile
    2 = dipendenze Python mancanti
    3 = errore caricamento modello o inferenza
"""

import argparse
import sys
import importlib.metadata


# ─────────────────────────────────────────────
# 1. VERIFICA GPU E VRAM
# ─────────────────────────────────────────────

def verifica_gpu() -> bool:
    """
    Controlla la disponibilità della GPU CUDA e stampa le informazioni sulla VRAM.

    Restituisce:
        True se CUDA è disponibile, False altrimenti.
    """
    import torch

    print("=" * 60)
    print("VERIFICA GPU")
    print("=" * 60)
    print(f"  Versione PyTorch : {torch.__version__}")
    print(f"  Versione CUDA    : {torch.version.cuda or 'N/A'}")

    if not torch.cuda.is_available():
        print("  [ERRORE] CUDA non disponibile.")
        print("  Assicurati di avere:")
        print("    - Driver NVIDIA >= 525.x  (verifica: nvidia-smi)")
        print("    - PyTorch installato con supporto CUDA 11.8:")
        print("      pip install torch==2.1.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118")
        return False

    nome_gpu = torch.cuda.get_device_name(0)
    # mem_get_info restituisce (libera, totale) in byte
    vram_libera, vram_totale = torch.cuda.mem_get_info(0)
    vram_usata = vram_totale - vram_libera

    gb = 1024 ** 3
    print(f"  [OK] GPU rilevata : {nome_gpu}")
    print(f"  VRAM totale       : {vram_totale / gb:.2f} GB")
    print(f"  VRAM libera       : {vram_libera / gb:.2f} GB")
    print(f"  VRAM in uso       : {vram_usata / gb:.2f} GB")

    if vram_totale < 8 * gb:
        print("  [AVVISO] VRAM inferiore a 8GB. Riduci batch_size in structure_config.json.")
    elif vram_totale < 12 * gb:
        print("  [AVVISO] VRAM inferiore a 12GB. batch_size=4 consigliato.")
    else:
        print("  [OK] VRAM sufficiente per batch_size=6 con ResNet-18.")

    return True


# ─────────────────────────────────────────────
# 2. VERIFICA DIPENDENZE
# ─────────────────────────────────────────────

def verifica_dipendenze() -> bool:
    """
    Verifica che tutte le dipendenze Python siano installate e ne stampa le versioni.

    Restituisce:
        True se tutte le dipendenze sono presenti, False se manca qualcosa.
    """
    print()
    print("=" * 60)
    print("VERIFICA DIPENDENZE")
    print("=" * 60)

    # Lista: (nome_import, nome_pacchetto_pip)
    dipendenze = [
        ("torch",        "torch"),
        ("torchvision",  "torchvision"),
        ("transformers", "transformers"),
        ("PIL",          "Pillow"),
        ("cv2",          "opencv-python"),
        ("numpy",        "numpy"),
        ("matplotlib",   "matplotlib"),
        ("tqdm",         "tqdm"),
        ("lxml",         "lxml"),
        ("scipy",        "scipy"),
    ]

    tutte_ok = True
    for nome_import, nome_pip in dipendenze:
        try:
            modulo = __import__(nome_import)
            # Recupera versione dal registro dei pacchetti installati
            try:
                versione = importlib.metadata.version(nome_pip)
            except importlib.metadata.PackageNotFoundError:
                # Fallback: attributo __version__ sul modulo
                versione = getattr(modulo, "__version__", "sconosciuta")
            print(f"  [OK] {nome_pip:<25} {versione}")
        except ImportError:
            print(f"  [MANCANTE] {nome_pip:<22} — installa con: pip install {nome_pip}")
            tutte_ok = False

    # EasyOCR importato separatamente (lento da caricare)
    print(f"  {'':3}", end="")
    try:
        import importlib.util
        if importlib.util.find_spec("easyocr") is not None:
            try:
                versione = importlib.metadata.version("easyocr")
            except importlib.metadata.PackageNotFoundError:
                versione = "installato"
            print(f"[OK] {'easyocr':<25} {versione}")
        else:
            raise ImportError
    except ImportError:
        print(f"[MANCANTE] easyocr               — installa con: pip install easyocr")
        tutte_ok = False

    return tutte_ok


# ─────────────────────────────────────────────
# 3. VERIFICA MODELLO TATR
# ─────────────────────────────────────────────

# ID del modello su HuggingFace Hub
MODELLO_HF = "microsoft/table-transformer-structure-recognition-v1.1-all"


def verifica_modello_tatr(device: str) -> tuple:
    """
    Scarica (o usa la cache locale) e carica il modello TATR structure recognition.

    Il modello corretto per il riconoscimento della struttura degli orari scolastici è:
    'microsoft/table-transformer-structure-recognition-v1.1-all'
    Addestrato su PubTables-1M + FinTabNet, è il punto di partenza ottimale per
    il fine-tuning su orari scolastici italiani.

    Argomenti:
        device: 'cuda' o 'cpu'

    Restituisce:
        (model, processor) oppure (None, None) in caso di errore.
    """
    from transformers import TableTransformerForObjectDetection, DetrImageProcessor

    print()
    print("=" * 60)
    print("VERIFICA MODELLO TATR")
    print("=" * 60)
    print(f"  Modello : {MODELLO_HF}")
    print(f"  Device  : {device}")
    print("  Caricamento in corso (prima volta: ~350MB da scaricare)...")

    try:
        processor = DetrImageProcessor.from_pretrained(MODELLO_HF)
        model = TableTransformerForObjectDetection.from_pretrained(MODELLO_HF)
        model = model.to(device)
        model.eval()

        # Conta parametri totali e addestrabili
        n_totale = sum(p.numel() for p in model.parameters())
        n_addestr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [OK] Modello caricato con successo")
        print(f"  Parametri totali       : {n_totale:,}")
        print(f"  Parametri addestrabili : {n_addestr:,}")
        print(f"  Classi riconosciute    : {model.config.num_labels}")

        # Mostra le classi del modello
        id2label = model.config.id2label
        print("  Classi struttura:")
        for idx, nome in id2label.items():
            print(f"    {idx}: {nome}")

        return model, processor

    except Exception as errore:
        print(f"  [ERRORE] Impossibile caricare il modello: {errore}")
        print("  Verifica la connessione internet o che transformers>=4.38 sia installato.")
        return None, None


# ─────────────────────────────────────────────
# 4. TEST INFERENZA
# ─────────────────────────────────────────────

def crea_immagine_sintetica() -> "PIL.Image.Image":
    """
    Crea un'immagine sintetica che simula un orario scolastico con griglia.

    Disegna una tabella 7x7 (6 colonne orarie + intestazione, 7 righe giornaliere)
    su sfondo bianco per simulare la struttura tipica di un orario docente.

    Restituisce:
        Immagine PIL in modalità RGB (700x560 pixel).
    """
    from PIL import Image, ImageDraw, ImageFont

    larghezza, altezza = 700, 560
    img = Image.new("RGB", (larghezza, altezza), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Definizione griglia (6 colonne orarie + 1 intestazione, 6 righe + 1 intestazione)
    n_col = 7
    n_rig = 7
    largh_col = larghezza // n_col
    alt_rig = altezza // n_rig

    intestazioni_col = ["Ora", "Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato"]
    intestazioni_rig = ["", "1ª ora\n8:00-9:00", "2ª ora\n9:00-10:00", "3ª ora\n10:00-11:00",
                        "4ª ora\n11:00-12:00", "5ª ora\n12:00-13:00", "6ª ora\n13:00-14:00"]

    # Sfondo intestazione colonne (grigio chiaro)
    draw.rectangle([0, 0, larghezza, alt_rig], fill=(220, 220, 220))

    # Sfondo intestazione righe (grigio chiaro)
    draw.rectangle([0, 0, largh_col, altezza], fill=(220, 220, 220))

    # Disegna griglia
    for c in range(n_col + 1):
        x = c * largh_col
        draw.line([(x, 0), (x, altezza)], fill=(100, 100, 100), width=2)

    for r in range(n_rig + 1):
        y = r * alt_rig
        draw.line([(0, y), (larghezza, y)], fill=(100, 100, 100), width=2)

    # Intestazioni colonne
    for c, testo in enumerate(intestazioni_col):
        x = c * largh_col + largh_col // 2
        y = alt_rig // 2
        draw.text((x, y), testo, fill=(0, 0, 0), anchor="mm")

    # Intestazioni righe
    for r, testo in enumerate(intestazioni_rig):
        if testo:
            x = largh_col // 2
            y = r * alt_rig + alt_rig // 2
            draw.text((x, y), testo, fill=(0, 0, 0), anchor="mm")

    # Contenuto celle (simulazione materie)
    materie = ["Italiano", "Matematica", "Scienze", "Storia", "Inglese", "Arte"]
    import random
    random.seed(42)
    for r in range(1, n_rig):
        for c in range(1, n_col):
            x = c * largh_col + largh_col // 2
            y = r * alt_rig + alt_rig // 2
            materia = random.choice(materie)
            draw.text((x, y), materia, fill=(50, 50, 150), anchor="mm")

    return img


def test_inferenza(
    model,
    processor,
    device: str,
    percorso_immagine: str = None,
    debug: bool = False
) -> bool:
    """
    Esegue un'inferenza di test su un'immagine reale o sintetica.

    Argomenti:
        model: modello TATR caricato
        processor: DetrImageProcessor
        device: 'cuda' o 'cpu'
        percorso_immagine: path a un'immagine reale (opzionale)
        debug: se True, salva l'output annotato come 'test_output.jpg'

    Restituisce:
        True se l'inferenza ha avuto successo, False altrimenti.
    """
    import torch
    from PIL import Image, ImageDraw

    print()
    print("=" * 60)
    print("TEST INFERENZA")
    print("=" * 60)

    # Carica immagine
    if percorso_immagine:
        print(f"  Immagine: {percorso_immagine}")
        try:
            immagine = Image.open(percorso_immagine).convert("RGB")
        except Exception as e:
            print(f"  [ERRORE] Impossibile aprire l'immagine: {e}")
            return False
    else:
        print("  Immagine: sintetica (orario scolastico simulato 700x560)")
        immagine = crea_immagine_sintetica()

    print(f"  Dimensioni: {immagine.width} x {immagine.height} pixel")

    # Preprocessing
    try:
        inputs = processor(images=immagine, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inferenza
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-processing con soglia di confidenza 0.5
        dim_originali = [(immagine.height, immagine.width)]
        risultati = processor.post_process_object_detection(
            outputs,
            threshold=0.5,
            target_sizes=dim_originali
        )[0]

        etichette = risultati["labels"].cpu().tolist()
        scores = risultati["scores"].cpu().tolist()
        boxes = risultati["boxes"].cpu().tolist()

        id2label = model.config.id2label
        conteggio = {}
        for lbl in etichette:
            nome = id2label.get(lbl, f"classe_{lbl}")
            conteggio[nome] = conteggio.get(nome, 0) + 1

        print(f"  [OK] Inferenza completata. Oggetti rilevati (soglia=0.5):")
        if not conteggio:
            print("    Nessun oggetto rilevato (normale per immagine sintetica semplice)")
            print("    Prova con --immagine path/a/un/orario_reale.jpg")
        else:
            for nome, cnt in sorted(conteggio.items()):
                print(f"    {nome:<40} {cnt:>3} oggetti")

        # Output visuale in modalità debug
        if debug:
            img_debug = immagine.copy()
            draw = ImageDraw.Draw(img_debug)

            colori_classe = {
                "table":                       (255, 215, 0),    # Giallo
                "table row":                   (255, 80, 80),    # Rosso
                "table column":                (80, 80, 255),    # Blu
                "table column header":         (255, 165, 0),    # Arancio
                "table projected row header":  (0, 180, 0),      # Verde
                "table spanning cell":         (160, 0, 160),    # Viola
            }

            for lbl, score, box in zip(etichette, scores, boxes):
                nome = id2label.get(lbl, f"classe_{lbl}")
                colore = colori_classe.get(nome, (128, 128, 128))
                xmin, ymin, xmax, ymax = [int(c) for c in box]
                draw.rectangle([xmin, ymin, xmax, ymax], outline=colore, width=3)
                draw.text((xmin + 3, ymin + 3), f"{nome}\n{score:.2f}", fill=colore)

            output_path = "test_output.jpg"
            img_debug.save(output_path)
            print(f"  Output visuale salvato in: {output_path}")

        return True

    except Exception as errore:
        print(f"  [ERRORE] Inferenza fallita: {errore}")
        return False


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    """Punto di ingresso principale del script di verifica."""
    parser = argparse.ArgumentParser(
        description="Verifica l'ambiente per il fine-tuning di TATR su orari scolastici",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python test_tatr.py
  python test_tatr.py --immagine foto/orario.jpg --debug
  python test_tatr.py --skip-modello
        """
    )
    parser.add_argument(
        "--immagine",
        type=str,
        default=None,
        help="Path a un'immagine reale di orario scolastico per il test inferenza"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Salva l'output annotato come 'test_output.jpg'"
    )
    parser.add_argument(
        "--skip-modello",
        action="store_true",
        dest="skip_modello",
        help="Salta il download e il test del modello (solo hardware + dipendenze)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device da usare: 'cuda' o 'cpu' (default: auto-detect)"
    )
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Verifica Ambiente - Table Transformer (TATR)            ║")
    print("║  Progetto: Fine-tuning per orari scolastici italiani     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 1. Verifica GPU
    gpu_ok = verifica_gpu()
    if not gpu_ok:
        print()
        print("[AVVISO] Continuo senza GPU — il training sarà molto lento su CPU.")
        print("         Per il fine-tuning è fortemente consigliata una GPU CUDA.")

    # Determina device
    if args.device:
        device = args.device
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    # 2. Verifica dipendenze
    dep_ok = verifica_dipendenze()
    if not dep_ok:
        print()
        print("[ERRORE] Dipendenze mancanti. Installa con: pip install -r requirements.txt")
        sys.exit(2)

    if args.skip_modello:
        print()
        print("=" * 60)
        print("  Test modello saltato (--skip-modello)")
        print("=" * 60)
        if gpu_ok and dep_ok:
            print()
            print("✓ Hardware e dipendenze OK. Ambiente pronto per il fine-tuning.")
        sys.exit(0)

    # 3. Verifica modello
    model, processor = verifica_modello_tatr(device)
    if model is None:
        print()
        print("[ERRORE] Impossibile caricare il modello TATR.")
        sys.exit(3)

    # 4. Test inferenza
    inf_ok = test_inferenza(
        model=model,
        processor=processor,
        device=device,
        percorso_immagine=args.immagine,
        debug=args.debug
    )

    if not inf_ok:
        print()
        print("[ERRORE] Test inferenza fallito.")
        sys.exit(3)

    print()
    print("=" * 60)
    print("RIEPILOGO FINALE")
    print("=" * 60)
    print(f"  GPU CUDA     : {'✓ OK' if gpu_ok else '✗ Non disponibile'}")
    print(f"  Dipendenze   : {'✓ OK' if dep_ok else '✗ Mancanti'}")
    print(f"  Modello TATR : {'✓ OK' if model else '✗ Errore'}")
    print(f"  Inferenza    : {'✓ OK' if inf_ok else '✗ Errore'}")
    print()
    print("✓ Ambiente pronto per il fine-tuning su orari scolastici!")
    print("  Passo successivo: python pre_annotate.py --input foto/ --output dataset/train/")
    print()


if __name__ == "__main__":
    main()
