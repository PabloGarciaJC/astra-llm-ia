# Astra LLM AI — Transformer desde cero

Stack completo para construir y entrenar tu propio modelo de lenguaje (LLM) desde cero, en tu máquina, sin servicios externos ni coste por consulta. 100% local con Docker.

---

## ¿Qué tecnologías usa este proyecto y cómo?

Un LLM moderno se compone de varias capas. Aquí se explica qué tiene este proyecto en cada una:

### 1. Núcleo — Deep Learning + Transformers

**Lo que usa este proyecto:** implementación completa del Transformer escrita desde cero en PyTorch.

Sin esto no hay LLM. Este proyecto lo implementa manualmente en `train.py`:

| Componente | Qué hace |
|---|---|
| Token Embedding | Convierte cada carácter en un vector numérico |
| Positional Embedding | Añade la posición de cada token en la secuencia |
| Multi-Head Self-Attention | Aprende relaciones entre tokens (causal/enmascarado) |
| Feed Forward Network | Procesa la información después de la atención |
| Layer Norm + Residual | Estabiliza el entrenamiento |
| Linear Head | Predice el siguiente token |

```
Texto de entrada
    ↓
Tokenizer          → convierte caracteres en números
    ↓
Token Embedding    → convierte números en vectores
    ↓
Positional Embed.  → añade posición de cada token
    ↓
[Bloque × N_LAYER]
  ├── Multi-Head Attention   → aprende relaciones entre tokens
  ├── Feed Forward           → procesa la información
  └── Layer Norm + Residual  → estabiliza el entrenamiento
    ↓
Linear Head        → predice el siguiente token
    ↓
Texto generado
```

---

### 2. Framework de entrenamiento — PyTorch

**Lo que usa este proyecto:** PyTorch como único framework.

| Herramienta PyTorch | Uso en el proyecto |
|---|---|
| `torch.nn.Module` | Base de la arquitectura del Transformer |
| `torch.optim.AdamW` | Optimizador con decaimiento de pesos |
| `torch.utils.data.DataLoader` | Carga de batches durante el entrenamiento |
| `torch.nn.utils.clip_grad_norm_` | Estabiliza el gradiente (evita explosión) |
| `torch.cuda` | Detecta automáticamente si hay GPU disponible |

> Este proyecto no usa TensorFlow ni JAX. Solo PyTorch. Esto es lo habitual: se elige un framework principal y se usa de forma consistente.

---

### 3. Hardware — CPU o GPU (una sola máquina)

**Lo que usa este proyecto:** una sola máquina con CPU o GPU.

```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

El entrenamiento se ejecuta automáticamente en GPU si está disponible, o en CPU si no. **No usa entrenamiento distribuido** (no hay miles de GPUs en paralelo como en GPT-4 o LLaMA). Esto es correcto para un modelo educativo o experimental en local.

> Para activar GPU añade `--gpus all` al servicio `llm-trainer` en `docker-compose.yml`.

---

### 4. Datos — Corpus propio en texto plano

**Lo que usa este proyecto:** un único archivo de texto como corpus de entrenamiento.

- **Archivo:** `llm-trainer/data/corpus.txt`
- **Tokenizer:** nivel carácter (cada carácter es un token)
- **Split:** 90% entrenamiento / 10% validación

El modelo aprende únicamente del texto que tú le pongas en `corpus.txt`. A diferencia de modelos como GPT, no se mezclan webs, libros y código — aquí el corpus lo defines tú.

> El tokenizer de nivel carácter es simple y educativo. Los LLM de producción usan tokenizers subword (BPE/WordPiece) que manejan vocabularios de 30.000–100.000 tokens.

---

### 5. Técnicas de entrenamiento — Preentrenamiento

**Lo que usa este proyecto:** solo preentrenamiento (desde cero).

| Etapa | ¿Este proyecto la usa? | Descripción |
|---|---|---|
| **Preentrenamiento** | ✅ Sí | El modelo aprende a predecir el siguiente carácter en el corpus |
| **Fine-tuning** | ❌ No | Ajuste del modelo para una tarea específica |
| **RLHF** | ❌ No | Aprendizaje por refuerzo con feedback humano (usado en ChatGPT) |

El preentrenamiento es la base de todo LLM: el modelo aprende la estructura del lenguaje prediciendo el siguiente token. Las etapas siguientes (fine-tuning, RLHF) se aplican encima de un modelo ya preentrenado.

---

## Stack de IA — resumen

| Capa | Tecnología | Rol en este proyecto |
|---|---|---|
| Núcleo | **Transformer** (PyTorch) | Arquitectura del LLM implementada desde cero |
| Framework | **PyTorch** | Entrenamiento, optimización, inferencia |
| Tokenizer | **Nivel carácter** | Convierte el corpus en tokens entrenables |
| Corpus | **corpus.txt** | Texto con el que aprende el modelo |
| Técnica | **Preentrenamiento** | El modelo aprende a predecir el siguiente token |
| Hardware | **CPU / GPU** | Una sola máquina (local) |
| Runtime | **Docker** | Entorno reproducible y aislado |
| Interfaz | **Flask + HTML** | Web para generar texto con el modelo entrenado |

---

## Instalación

### Requisitos previos

- **Docker** y **Docker Compose** instalados
- **Make** para automatizar los comandos

### Pasos

1. Clona el repositorio.
2. Ejecuta el setup inicial:

```bash
make init-app
```

3. Añade tu texto de entrenamiento en `llm-trainer/data/corpus.txt`.
4. Lanza el entrenamiento:

```bash
make train
```

5. Abre la interfaz web:

```bash
make web
# → http://localhost:8080
```

---

## Comandos disponibles

### Entrenamiento

| Comando | Descripción |
|---|---|
| `make train` | Lanza el entrenamiento del LLM |
| `make train-logs` | Ver los logs del trainer en tiempo real |
| `make train-shell` | Accede al shell del contenedor trainer |

### Generación desde terminal

```bash
make generate PROMPT="Había una vez" GEN_LEN=300 TEMPERATURE=0.8
```

### Docker general

| Comando | Descripción |
|---|---|
| `make init-app` | Setup inicial |
| `make up` | Levanta todos los servicios |
| `make down` | Detiene los contenedores |
| `make build` | Reconstruye las imágenes sin caché |
| `make logs` | Ver logs de todos los servicios |
| `make ps` | Estado de los contenedores |

---

## Hiperparámetros configurables

Se configuran en el archivo `.env`:

| Variable | Por defecto | Descripción |
|---|---|---|
| `EPOCHS` | 50 | Número de épocas de entrenamiento |
| `BATCH_SIZE` | 32 | Tamaño del lote |
| `LEARNING_RATE` | 3e-4 | Tasa de aprendizaje |
| `BLOCK_SIZE` | 64 | Longitud del contexto (tokens) |
| `N_EMBED` | 128 | Dimensión de los embeddings |
| `N_HEAD` | 4 | Número de cabezales de atención |
| `N_LAYER` | 4 | Número de bloques transformer |
| `DROPOUT` | 0.1 | Tasa de dropout |

---

## Estructura del proyecto

```
llm-local-AI/
├── docker-compose.yml        # Orquestación de servicios
├── Makefile                  # Comandos automatizados
├── .env                      # Variables de entorno (local)
├── .env.example              # Plantilla de variables
├── llm-trainer/
│   ├── Dockerfile            # Imagen Python + PyTorch
│   ├── requirements.txt      # Dependencias Python
│   ├── train.py              # Transformer + training loop
│   ├── generate.py           # Generación de texto por terminal
│   ├── app.py                # Interfaz web (Flask)
│   ├── data/
│   │   └── corpus.txt        # Texto de entrenamiento
│   ├── output/
│   │   └── model.pt          # Modelo entrenado (se genera)
│   └── templates/
│       └── index.html        # UI web
└── scripts/
    └── docker-bash-trainer.sh
```

---

## Requisitos mínimos

| Componente | Mínimo |
|---|---|
| RAM | 4 GB |
| Disco | 5 GB libres |
| Docker | 24+ |
| Python | 3.11 (vía Docker) |

---

## Contáctame / Sígueme en mis redes sociales

| Red Social | Descripción | Enlace |
|------------|-------------|--------|
| **Facebook** | Conéctate y mantente al tanto de mis actualizaciones. | [Presiona aquí](https://www.facebook.com/PabloGarciaJC) |
| **YouTube** | Fundamentos de la programación, tutoriales y noticias. | [Presiona aquí](https://www.youtube.com/@pablogarciajc) |
| **Página Web** | Más información sobre mis proyectos y servicios. | [Presiona aquí](https://pablogarciajc.com/) |
| **LinkedIn** | Sigue mi carrera profesional y establece conexiones. | [Presiona aquí](https://www.linkedin.com/in/pablogarciajc) |
| **Instagram** | Fotos, proyectos y contenido relacionado. | [Presiona aquí](https://www.instagram.com/pablogarciajc) |
| **Twitter** | Proyectos, pensamientos y actualizaciones. | [Presiona aquí](https://x.com/PabloGarciaJC) |

---
> _"La inteligencia artificial no reemplaza al ser humano, lo potencia."_
