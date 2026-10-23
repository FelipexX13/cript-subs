# -*- coding: utf-8 -*-
"""
Cript Subs - Escucha el audio del PC, lo transcribe en local con Whisper (GPU)
y va tomando notas/resumen con un modelo local de Ollama.

Uso rapido:
    .\.venv\Scripts\python.exe escuchar.py
    .\.venv\Scripts\python.exe escuchar.py --fuente microfono --modelo large-v3
    .\.venv\Scripts\python.exe escuchar.py --listar-dispositivos

Para parar: Ctrl+C  (al salir genera un resumen final).
"""
import os
import sys
import glob
import time
import queue
import threading
import argparse
import datetime as dt


# --------------------------------------------------------------------------
# 1. Activar las DLLs de CUDA (vienen de los paquetes pip nvidia-*-cu12)
# --------------------------------------------------------------------------
def activar_cuda():
    bindirs = glob.glob(
        os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "*", "bin")
    )
    for d in bindirs:
        try:
            os.add_dll_directory(d)
        except OSError:
            pass
    if bindirs:
        os.environ["PATH"] = os.pathsep.join(bindirs) + os.pathsep + os.environ.get("PATH", "")


activar_cuda()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import requests
import soundcard as sc
from faster_whisper import WhisperModel


# --------------------------------------------------------------------------
# Constantes de audio
# --------------------------------------------------------------------------
SAMPLERATE = 16000          # Whisper trabaja a 16 kHz mono
BLOCK_SEG = 0.5             # tamano de cada bloque que graba el hilo de audio
BLOCK_FRAMES = int(SAMPLERATE * BLOCK_SEG)


# --------------------------------------------------------------------------
# 2. Hilo grabador: captura bloques de audio y los mete en una cola
# --------------------------------------------------------------------------
class Grabador(threading.Thread):
    def __init__(self, micro, cola, parar_evt):
        super().__init__(daemon=True)
        self.micro = micro
        self.cola = cola
        self.parar_evt = parar_evt

    def run(self):
        # Bucle resistente: si el dispositivo da un hipo (p. ej. un Bluetooth que
        # se desconecta un instante) reabrimos el flujo en vez de morir callados.
        while not self.parar_evt.is_set():
            try:
                with self.micro.recorder(samplerate=SAMPLERATE, channels=1,
                                         blocksize=BLOCK_FRAMES) as rec:
                    while not self.parar_evt.is_set():
                        data = rec.record(numframes=BLOCK_FRAMES)   # (n, 1) float32
                        self.cola.put(data[:, 0].copy())
            except Exception as e:
                if self.parar_evt.is_set():
                    break
                print(f"\n[AVISO] Problema con el audio ({e}). Reintentando en 2 s...")
                time.sleep(2)


def rms(bloque):
    return float(np.sqrt(np.mean(bloque.astype(np.float64) ** 2))) if bloque.size else 0.0


# --------------------------------------------------------------------------
# 3. Cliente Ollama (HTTP local)
# --------------------------------------------------------------------------
class Ollama:
    def __init__(self, modelo, url="http://localhost:11434", num_ctx=8192):
        self.modelo = modelo
        self.url = url.rstrip("/")
        self.num_ctx = num_ctx   # contexto amplio para resumir reuniones largas

    def generar(self, prompt, system=None, timeout=180):
        payload = {"model": self.modelo, "prompt": prompt, "stream": False,
                   "options": {"temperature": 0.2, "num_ctx": self.num_ctx}}
        if system:
            payload["system"] = system
        r = requests.post(f"{self.url}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def disponible(self):
        try:
            requests.get(f"{self.url}/api/tags", timeout=5).raise_for_status()
            return True
        except Exception:
            return False


SYSTEM_NOTAS = (
    "Eres un asistente que toma notas de un audio/reunion en curso. "
    "Recibes el ultimo fragmento transcrito. Extrae SOLO lo relevante en vinetas "
    "muy concisas en espanol: ideas clave, decisiones, datos, tareas o preguntas. "
    "No repitas literalmente, sintetiza. Si el fragmento no aporta nada, responde exactamente: -"
)

SYSTEM_RESUMEN = (
    "Eres un asistente que resume transcripciones. A partir del material proporcionado, "
    "devuelve en espanol y en Markdown: un parrafo de Resumen, una lista de Puntos clave, "
    "una lista de Decisiones (si las hay) y una lista de Tareas / acciones pendientes "
    "(con responsable si se menciona). Se claro y conciso."
)


def resumir_jerarquico(ollama, texto, max_chars=7000):
    """Resumen seguro para textos largos (reuniones de horas).

    Si cabe en una pasada, resume directo. Si no, trocea, resume cada parte y
    luego combina (map-reduce), repitiendo hasta que quepa. Asi el resumen final
    nunca se ve recortado por el limite de contexto del modelo.
    """
    texto = texto.strip()
    if len(texto) <= max_chars:
        return ollama.generar(f"Material:\n\"\"\"\n{texto}\n\"\"\"", SYSTEM_RESUMEN, timeout=300)

    # MAP: resumir cada trozo a vinetas
    trozos = [texto[i:i + max_chars] for i in range(0, len(texto), max_chars)]
    print(f"    (resumen largo: {len(trozos)} partes)")
    parciales = []
    for n, t in enumerate(trozos, 1):
        p = ollama.generar(
            f"Resume en vinetas concisas en espanol las ideas, decisiones y tareas de esta "
            f"parte ({n}/{len(trozos)}) de una reunion:\n\"\"\"\n{t}\n\"\"\"",
            SYSTEM_NOTAS, timeout=300)
        if p and p.strip() != "-":
            parciales.append(p)
    combinado = "\n".join(parciales)
    # REDUCE: si el combinado sigue siendo enorme, recursar; si no, resumen final
    if len(combinado) > max_chars:
        return resumir_jerarquico(ollama, combinado, max_chars)
    return ollama.generar(f"Material:\n\"\"\"\n{combinado}\n\"\"\"", SYSTEM_RESUMEN, timeout=300)


# --------------------------------------------------------------------------
# 4. Seleccion de dispositivos de audio
# --------------------------------------------------------------------------
def listar_dispositivos():
    print("\n=== ALTAVOCES (captura del sistema / loopback) ===")
    for s in sc.all_speakers():
        marca = "  <- predeterminado" if s.name == sc.default_speaker().name else ""
        print(f"  - {s.name}{marca}")
    print("\n=== MICROFONOS ===")
    for m in sc.all_microphones():
        marca = "  <- predeterminado" if m.name == sc.default_microphone().name else ""
        print(f"  - {m.name}{marca}")
    print()


def micro_sistema():
    """Micro 'loopback' del altavoz predeterminado = lo que suena en el PC."""
    spk = sc.default_speaker()
    return sc.get_microphone(spk.name, include_loopback=True)


def micro_microfono():
    return sc.default_microphone()


# --------------------------------------------------------------------------
# 5. Programa principal
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Transcribe el audio del PC y toma notas en local.")
    ap.add_argument("--fuente", choices=["sistema", "microfono", "ambos"], default="sistema",
                    help="Que audio capturar (def: sistema = lo que suena en el PC)")
    ap.add_argument("--modelo", default="medium",
                    help="Modelo Whisper: tiny/base/small/medium/large-v3 (def: medium)")
    ap.add_argument("--idioma", default="es", help="Idioma (def: es; usa 'auto' para detectar)")
    ap.add_argument("--dispositivo", choices=["cuda", "cpu"], default="cuda",
                    help="Donde corre Whisper (def: cuda)")
    ap.add_argument("--ollama-modelo", default="qwen2.5:7b",
                    help="Modelo de Ollama para las notas (def: qwen2.5:7b)")
    ap.add_argument("--intervalo-resumen", type=int, default=120,
                    help="Cada cuantos segundos actualiza las notas (def: 120)")
    ap.add_argument("--solo-transcribir", action="store_true",
                    help="No usa Ollama, solo guarda la transcripcion")
    ap.add_argument("--min-seg", type=float, default=6.0, help="Duracion minima de un fragmento")
    ap.add_argument("--max-seg", type=float, default=20.0, help="Duracion maxima de un fragmento")
    ap.add_argument("--umbral-silencio", type=float, default=0.012,
                    help="RMS por debajo del cual se considera silencio/pausa")
    ap.add_argument("--listar-dispositivos", action="store_true")
    args = ap.parse_args()

    if args.listar_dispositivos:
        listar_dispositivos()
        return

    # --- carpetas de salida ---
    base = os.path.dirname(os.path.abspath(__file__))
    carpeta_notas = os.path.join(base, "notas")
    os.makedirs(carpeta_notas, exist_ok=True)
    sello = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_trans = os.path.join(carpeta_notas, f"transcripcion_{sello}.txt")
    ruta_notas = os.path.join(carpeta_notas, f"notas_{sello}.md")

    # --- Ollama ---
    ollama = None
    if not args.solo_transcribir:
        ollama = Ollama(args.ollama_modelo)
        if not ollama.disponible():
            print("[AVISO] No se pudo contactar a Ollama en localhost:11434. "
                  "Continuo solo con transcripcion.")
            ollama = None

    # --- modelo Whisper ---
    print(f"[*] Cargando Whisper '{args.modelo}' en {args.dispositivo}...")
    compute = "float16" if args.dispositivo == "cuda" else "int8"
    try:
        modelo = WhisperModel(args.modelo, device=args.dispositivo, compute_type=compute)
    except Exception as e:
        print(f"[AVISO] Fallo en {args.dispositivo} ({e}). Reintento en CPU (int8)...")
        modelo = WhisperModel(args.modelo, device="cpu", compute_type="int8")
        args.dispositivo = "cpu"
    print("[*] Modelo listo.")

    idioma = None if args.idioma.lower() == "auto" else args.idioma

    # --- montar grabadores ---
    parar = threading.Event()
    hilos = []
    if args.fuente in ("sistema", "ambos"):
        q_sys = queue.Queue()
        hilos.append(Grabador(micro_sistema(), q_sys, parar))
    else:
        q_sys = None
    if args.fuente in ("microfono", "ambos"):
        q_mic = queue.Queue()
        hilos.append(Grabador(micro_microfono(), q_mic, parar))
    else:
        q_mic = None
    for h in hilos:
        h.start()

    print("\n" + "=" * 60)
    print(f"  ESCUCHANDO ({args.fuente}). Transcripcion -> {os.path.basename(ruta_trans)}")
    if ollama:
        print(f"  Notas con Ollama '{args.ollama_modelo}' -> {os.path.basename(ruta_notas)}")
    print("  Pulsa Ctrl+C para parar y generar el resumen final.")
    print("=" * 60 + "\n")

    buffer = []                 # lista de bloques np
    dur_buffer = 0.0
    transcripcion_total = []    # todas las lineas transcritas
    texto_para_notas = []       # texto nuevo aun no resumido
    notas_acumuladas = []       # todas las notas ya generadas (base del resumen final)
    ultimo_resumen = time.time()

    f_trans = open(ruta_trans, "a", encoding="utf-8")

    def leer_bloque():
        """Devuelve un bloque mono (mezcla de fuentes si es 'ambos')."""
        if q_sys is not None and q_mic is not None:
            a = q_sys.get()
            b = q_mic.get()
            n = min(len(a), len(b))
            mez = a[:n] + b[:n]
            return np.clip(mez, -1.0, 1.0)
        elif q_sys is not None:
            return q_sys.get()
        else:
            return q_mic.get()

    def transcribir(audio):
        segmentos, _ = modelo.transcribe(
            audio, language=idioma, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
        )
        return " ".join(s.text.strip() for s in segmentos).strip()

    def actualizar_notas():
        nonlocal texto_para_notas
        if not ollama or not texto_para_notas:
            return
        nuevo = " ".join(texto_para_notas).strip()
        texto_para_notas = []
        if len(nuevo) < 20:
            return
        try:
            notas = ollama.generar(f"Fragmento transcrito:\n\"\"\"\n{nuevo}\n\"\"\"", SYSTEM_NOTAS)
        except Exception as e:
            print(f"[AVISO] Ollama fallo al generar notas: {e}")
            return
        if notas and notas.strip() not in ("-", ""):
            hora = dt.datetime.now().strftime("%H:%M:%S")
            notas_acumuladas.append(notas)
            with open(ruta_notas, "a", encoding="utf-8") as fn:
                fn.write(f"\n### {hora}\n{notas}\n")
            print(f"\n  [notas {hora}] actualizadas ->\n{notas}\n")

    try:
        while True:
            bloque = leer_bloque()
            buffer.append(bloque)
            dur_buffer += len(bloque) / SAMPLERATE

            pausa = rms(bloque) < args.umbral_silencio
            cortar = (dur_buffer >= args.min_seg and pausa) or dur_buffer >= args.max_seg
            if not cortar:
                continue

            audio = np.concatenate(buffer)
            buffer, dur_buffer = [], 0.0

            # si todo el fragmento es practicamente silencio, lo saltamos
            if rms(audio) < args.umbral_silencio * 0.7:
                continue

            texto = transcribir(audio)
            if not texto:
                continue

            hora = dt.datetime.now().strftime("%H:%M:%S")
            linea = f"[{hora}] {texto}"
            print(linea)
            f_trans.write(linea + "\n")
            f_trans.flush()
            transcripcion_total.append(texto)
            texto_para_notas.append(texto)

            if ollama and time.time() - ultimo_resumen >= args.intervalo_resumen:
                actualizar_notas()
                ultimo_resumen = time.time()

    except KeyboardInterrupt:
        print("\n[*] Parando...")
    finally:
        parar.set()
        f_trans.close()
        # ultimo lote de notas pendientes
        if ollama:
            actualizar_notas()

        # resumen final (a partir de las notas ya condensadas; si no hubiera,
        # de la transcripcion completa). resumir_jerarquico trocea si hace falta,
        # asi que aguanta reuniones de horas sin recortarse.
        if ollama and (notas_acumuladas or transcripcion_total):
            print("[*] Generando resumen final con Ollama...")
            base = "\n".join(notas_acumuladas) if notas_acumuladas else "\n".join(transcripcion_total)
            try:
                resumen = resumir_jerarquico(ollama, base)
                with open(ruta_notas, "a", encoding="utf-8") as fn:
                    fn.write("\n\n---\n\n## RESUMEN FINAL\n\n" + resumen + "\n")
                print("\n" + "=" * 60)
                print(resumen)
                print("=" * 60)
            except Exception as e:
                print(f"[AVISO] No se pudo generar el resumen final: {e}")

        print(f"\n[OK] Transcripcion: {ruta_trans}")
        if ollama:
            print(f"[OK] Notas/resumen: {ruta_notas}")


if __name__ == "__main__":
    main()
