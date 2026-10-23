# Cript Subs — Transcripción y notas del PC, 100% en local

Escucha el audio de tu PC (o el micrófono), lo **transcribe en tiempo real** con
Whisper acelerado por tu **GPU NVIDIA**, y va **tomando notas y un resumen** con un
modelo local de **Ollama**. Nada sale a internet.

## Requisitos

- Windows (la captura del audio del sistema usa loopback WASAPI) y Python 3.11.
- [Ollama](https://ollama.com) corriendo con algún modelo, p. ej. `ollama pull qwen2.5:7b`.
- GPU NVIDIA (probado en **RTX 5070 Ti / Blackwell**). Si la GPU falla, el script
  cae solo a CPU.

## Instalación

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Cómo usarlo

Lo más fácil: doble clic en **`Iniciar.bat`** (captura lo que suena en el PC).

O desde PowerShell, en esta carpeta:

```powershell
.\.venv\Scripts\python.exe escuchar.py
```

Para **parar**: `Ctrl + C`. Al salir genera un **resumen final**.

Los resultados se guardan en la carpeta **`notas\`**:
- `transcripcion_<fecha>.txt` — todo lo dicho, con marcas de hora.
- `notas_<fecha>.md` — notas periódicas + resumen final (Markdown).

## Opciones

```powershell
# Capturar el micrófono en vez del sistema
.\.venv\Scripts\python.exe escuchar.py --fuente microfono

# Capturar AMBOS (sistema + micro) — ideal para videollamadas
.\.venv\Scripts\python.exe escuchar.py --fuente ambos

# Más calidad (más lento, ~3 GB de VRAM la primera vez descarga el modelo)
.\.venv\Scripts\python.exe escuchar.py --modelo large-v3

# Ver los dispositivos de audio detectados
.\.venv\Scripts\python.exe escuchar.py --listar-dispositivos

# Solo transcribir, sin notas de Ollama
.\.venv\Scripts\python.exe escuchar.py --solo-transcribir

# Cambiar el modelo de Ollama que toma las notas
.\.venv\Scripts\python.exe escuchar.py --ollama-modelo llama3.1

# Ayuda completa
.\.venv\Scripts\python.exe escuchar.py --help
```

| Opción | Por defecto | Qué hace |
|---|---|---|
| `--fuente` | `sistema` | `sistema` / `microfono` / `ambos` |
| `--modelo` | `medium` | Modelo Whisper: `tiny` `base` `small` `medium` `large-v3` |
| `--idioma` | `es` | Idioma; usa `auto` para detectarlo |
| `--ollama-modelo` | `qwen2.5:7b` | Modelo local para las notas |
| `--intervalo-resumen` | `120` | Cada cuántos segundos actualiza las notas |
| `--dispositivo` | `cuda` | `cuda` o `cpu` (cae a CPU solo si falla la GPU) |

## Reuniones largas (1-2h o más)

Está preparado para aguantarlas:
- La transcripción se guarda en disco **en cada fragmento**, así que aunque algo
  falle a mitad, no pierdes lo ya transcrito.
- Cada ~2 min genera notas y las va anexando (no espera al final).
- El **resumen final** se hace de forma **jerárquica** (resume por partes y luego
  combina) y con contexto ampliado (`num_ctx=8192`), de modo que **no se recorta**
  aunque la reunión dure horas.
- La captura de audio se **reconecta sola** si el dispositivo da un hipo (p. ej. un
  altavoz Bluetooth que se desconecta un segundo).

No necesitas dividir la reunión manualmente: déjalo corriendo y para con `Ctrl+C`
al terminar.

## Notas técnicas

- La captura del audio del sistema usa **loopback WASAPI** (vía `soundcard`): no
  necesitas cables virtuales ni "Stereo Mix".
- El audio se trocea en pausas naturales (silencios) para no cortar frases.
- Whisper corre con `faster-whisper` (CTranslate2) en `float16` sobre la GPU.
- Las DLLs de CUDA se cargan desde los paquetes `nvidia-*-cu12` instalados en el venv
  (ver `activar_cuda()` en `escuchar.py`).
