# Guía de Refactorización — MoQ-NAS

> Generado: 2026-05-04  
> Estado: pendiente de implementación  
> Sesión de origen: análisis de modularidad completo

---

## Contexto del proyecto

MoQ-NAS es un framework de Neural Architecture Search (NAS) multi-objetivo. Implementa y compara algoritmos evolutivos cuánticos (QNAS, MO-QNAS) contra algoritmos genéticos clásicos (GA, NSGA-II, NSGA-III, MOEA/D). Evalúa redes CNN en múltiples objetivos: accuracy, tamaño del modelo, tiempo de inferencia y métricas de justicia demográfica (fairness).

**Punto de entrada principal:** `run_all_evolution.py`  
**Reentrenamiento:** `retrain_model.py` (individual), `retrain_parallel.py` (frente Pareto completo)  
**GPU requerida:** sí (NVIDIA, probado en L40S)  
**Sin tests automáticos** en el proyecto actualmente.

---

## Dos sistemas de configuración (ambos activos)

Esto es crítico para entender el proyecto:

| Carpeta | Propósito real | Extensión actual |
|---|---|---|
| `config_files/**/*.txt` | Configuración de **experimentos**: espacio de búsqueda (`function_dict`), hiperparámetros QNAS/GA, parámetros de entrenamiento | `.txt` con sintaxis YAML |
| `configs/*.yaml` | Metadatos de **datasets**: shape, num_classes, mean, std, task | `.yaml` |
| `configs/cfg_obj.json` | Sentido de optimización por objetivo (maximize/minimize) | `.json` |

Los archivos `.txt` de `config_files/` son YAML válido a pesar de la extensión. Son cargados por `core/config.py` mediante `yaml.safe_load()`. Un experimento típico referencia ambos sistemas: el `.txt` contiene `config_path_dataset: configs/cifar10.yaml`.

---

## Diagnóstico de problemas encontrados

### P1 — God Module: `utils/helpers.py` (1209 líneas)

Mezcla sin cohesión ~25 funciones de dominios completamente diferentes:

| Categoría | Funciones | Líneas |
|---|---|---|
| I/O YAML/JSON/Pickle | `load_yaml`, `save_pkl`, `load_pkl`, `update_yaml_file` | 39, 96, 111, 85 |
| Logging | `init_log` | 396 |
| Dataset | `download_dataset`, `setup_dataset_info`, `dataset_cache` | 596, 648, 636 |
| Gestión de experimentos | `check_files`, `load_evolved_data`, `load_log_params_evolution`, `calculate_time` | 345, 430, 529, 566 |
| Cache | `backup_cache`, `load_cache` | 713, 746 |
| Archivos/dirs | `delete_old_dirs`, `delete_old_dirs_v2`, `check_file_exists` | 325, 762, 145 |
| Visualización | `plot_training_history`, `plot_hypervolume_comparison`, `plot_pareto_evolution` | 219, 979, 1030 |
| Métricas offline | `compute_hypervolume_mixed` (standalone), `agg_results`, `test_acc_mean_std` | 852, 183, 176 |

**Impacto:** Importar `init_log` arrastra `matplotlib`, `plotly`, `pymoo`, `GPUtil`, `torchvision`, `medmnist`. Imposible testear funciones en aislamiento.

---

### P2 — Operadores MOEA duplicados en 3 lugares

| Función | Ubicaciones |
|---|---|
| `compute_hypervolume_mixed` | `utils/helpers.py:852`, `algorithms/ga/nsga2.py:223`, `algorithms/qnas/moqnas.py:360` |
| `dominates` | `algorithms/ga/nsga2.py:249`, `algorithms/qnas/moqnas.py:400` |
| `fast_nondominated_sort` | `algorithms/ga/nsga2.py:265`, `algorithms/qnas/moqnas.py:425` |
| `crowding_distance` | `algorithms/ga/nsga2.py:300`, `algorithms/qnas/moqnas.py:466` |
| `_simplex_lattice` | `algorithms/ga/nsga3.py:154`, `algorithms/ga/moead.py:53` |
| `_build_reference_directions` | `algorithms/ga/nsga3.py:137`, `algorithms/ga/moead.py:41` |
| `_to_minimization` | `algorithms/ga/nsga3.py:124`, `algorithms/ga/moead.py:66` |

**Divergencia ya detectada:** `dominates` en `nsga2.py:249` usa `objective_senses` dinámico correctamente; la versión en `moqnas.py:400` tiene la misma lógica pero diferente documentación. Ya hay riesgo de que una corrección se aplique en un lugar y no en el otro.

---

### P3 — `load_evolved_data` duplicada

- `utils/helpers.py:430` — función standalone, devuelve dict con `net`, `generation`, `individual`, `best_accuracy`
- `core/config.py:319` — método de clase, lógica idéntica de resolución de ruta, agrega `backbone_name` y `backbone_percentage`, guarda en `self.evolved_params`

Ambas buscan el symlink `best_so_far`, leen `training_params.txt` con la misma lógica. La versión de `config.py` es más completa. La de `helpers.py` es una versión incompleta/anterior.

---

### P4 — Rutas hardcodeadas

```python
# algorithms/ga/nsga2.py:48
with open("configs/cfg_obj.json", "r") as f:

# algorithms/qnas/moqnas.py:221
with open("configs/cfg_obj.json", "r") as f:
```

Ruta relativa que asume CWD = raíz del proyecto. Falla en notebooks, scripts ejecutados desde otro directorio, o cualquier test.

También: `TRAIN_TIMEOUT = 5400` hardcodeado en `core/cnn/trainer.py:27` sin documentación del por qué ese valor.

---

### P5 — Efectos secundarios en tiempo de import

```python
# core/cnn/master.py:23-29 — se ejecuta al hacer "import master"
project_root = os.getcwd()
log_directory = os.path.join(project_root, 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
log_file = os.path.join(log_directory, 'master.log')
LOGGER = init_log("INFO", name=__name__, file_path=log_file)
```

Mismo bloque repetido en `core/cnn/trainer.py:29-32`. Al importar estos módulos se crea el directorio `logs/` y el archivo `logs/master.log` en el CWD, sin que el usuario lo haya pedido.

---

### P6 — Ciclo de dependencias `core ↔ algorithms`

```python
# core/config.py:15 — core importa de algorithms
from algorithms.qnas.chromosome import QChromosomeNetwork, QChromosomeParams

# algorithms/ga/base_ga.py:7-8 — algorithms importa de core
from core import evaluation
from core import config as cfg
```

No se puede usar un algoritmo GA sin cargar `core.evaluation`. No se puede usar `core.config` sin instanciar el módulo de cromosomas QNAS. El grafo de dependencias tiene un ciclo.

---

### P7 — Naming confuso de carpetas de configuración

- `config_files/` suena a "archivos de config en general" pero contiene exclusivamente configs de experimentos
- Los archivos dentro son YAML válido con extensión `.txt` (sin highlighting en IDEs, sin autocompletado)
- `configs/` mezcla dataset YAMLs con `cfg_obj.json` (que es un registro de objetivos, no metadatos de dataset)

---

### P8 — `_Wrap` duplicada en `dataset_utils/factory.py`

La clase interna `_Wrap` (inyectar transform en un Subset de PyTorch) se define dos veces dentro de `build_datasets`:
- Línea 110: rama torchvision
- Línea 215: rama person/face binary

Código idéntico. Si se necesita agregar lógica (ej. cache), hay que modificar ambas.

---

### P9 — `old_files/` sin política de limpieza

9 archivos de debug/scripts obsoletos sin referencias desde el código en producción:
- `old_files/debug_*.py` — versiones antiguas de scripts de ejecución
- `old_files/fairness_baselines/*.py` — versiones antiguas de los scripts en `scripts/fairness_baseline/`
- `old_files/medmnist_inference_paper.py` — script puntual de inferencia

---

## Estructura de carpetas propuesta

```
moqnas/
├── algorithms/
│   ├── __init__.py
│   ├── pareto/                    ← NUEVO: operadores MOEA centralizados
│   │   ├── __init__.py
│   │   ├── dominance.py           # dominates(), fast_nondominated_sort()
│   │   ├── diversity.py           # crowding_distance()
│   │   ├── hypervolume.py         # compute_hypervolume_mixed()
│   │   └── reference_dirs.py     # simplex_lattice(), build_reference_directions(),
│   │                              # to_minimization()
│   ├── ga/
│   │   ├── __init__.py
│   │   ├── base_ga.py             # sin imports de core.*
│   │   ├── nsga2.py               # usa pareto.*; sin métodos duplicados
│   │   ├── nsga3.py               # usa pareto.*; sin _simplex_lattice, etc.
│   │   └── moead.py               # usa pareto.*
│   └── qnas/
│       ├── __init__.py
│       ├── chromosome.py
│       ├── population.py
│       ├── qnas2.py
│       ├── moqnas.py              # usa pareto.*; sin métodos duplicados
│       └── helpers/
│           ├── configs.py
│           ├── metrics_logger.py
│           ├── moea_helper.py
│           ├── operators.py
│           ├── rules.py
│           └── update_strategies.py
│
├── core/
│   ├── __init__.py
│   ├── config.py                  # sin import de algorithms.*
│   ├── evaluation.py
│   ├── cnn/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── model_resnet.py
│   │   ├── input.py
│   │   ├── trainer.py             # sin side-effects en módulo; TRAIN_TIMEOUT de settings
│   │   ├── master.py              # sin side-effects en módulo
│   │   ├── artifacts/
│   │   └── metrics/
│   └── fairness/
│
├── dataset_utils/
│   ├── __init__.py
│   ├── configs.py
│   ├── factory.py                 # _Wrap definida una sola vez a nivel de módulo
│   ├── sampling.py
│   ├── splits.py
│   ├── transformations.py
│   └── verify_dataset_splits.py
│
├── utils/
│   ├── __init__.py
│   ├── io.py                      ← NUEVO: load_yaml, save_pkl, load_pkl,
│   │                              #   load_evolved_data, backup_cache, load_cache,
│   │                              #   save_results_file, create_info_file,
│   │                              #   update_yaml_file, load_history_from_json,
│   │                              #   save_history_to_json, load_log_params_evolution,
│   │                              #   load_retrain_results, load_pareto_history
│   ├── logging_utils.py           ← NUEVO: init_log (solo esto)
│   ├── experiment.py              ← NUEVO: check_files, delete_old_dirs,
│   │                              #   delete_old_dirs_v2, calculate_time,
│   │                              #   natural_key, check_file_exists
│   ├── dataset.py                 ← NUEVO: download_dataset, setup_dataset_info,
│   │                              #   dataset_cache, _validate_dataset_info
│   ├── visualization.py           ← NUEVO: plot_training_history, agg_results,
│   │                              #   plot_confusion_matrix, test_acc_mean_std,
│   │                              #   get_gpu_memory, plot_hypervolume_over_epochs,
│   │                              #   _get_hypervolume_stats, plot_hypervolume_comparison,
│   │                              #   plot_pareto_evolution, load_data_for_pareto
│   └── helpers.py                 # facade temporal con re-exports (deprecar al final)
│
├── experiment_configs/            ← RENOMBRADO desde config_files/
│   ├── cifar/                     ← config_files_cifar/
│   ├── cifar_mo/                  ← config_files_cifar_mo/
│   ├── fairness/                  ← config_files_fairness/
│   ├── medmnist/                  ← config_files_med/
│   ├── medmnist_base/             ← config_files_med_base/
│   └── medmnist_mo/               ← config_files_med_mo/
│   (archivos .txt → .yaml)
│
├── dataset_configs/               ← RENOMBRADO desde configs/
│   ├── cifar10.yaml
│   ├── cifar100.yaml
│   ├── ... (todos los *.yaml actuales de configs/)
│   └── cfg_obj.json               ← movido desde configs/cfg_obj.json
│
├── settings.py                    ← NUEVO: constantes globales
│
├── run_all_evolution.py
├── retrain_model.py
├── retrain_parallel.py
├── scripts/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## Plan de implementación paso a paso

Los pasos 1–4 son independientes entre sí (cualquier orden). El paso 5 requiere que el paso 3 esté completo. El paso 6 es el más delicado y debe hacerse con el proyecto en estado estable.

---

### Paso 1 — Renombrar carpetas y extensiones de configuración

**Riesgo:** muy bajo | **Tiempo estimado:** 30 min

**Qué hacer:**

1. Renombrar carpeta: `config_files/` → `experiment_configs/`
2. Renombrar subcarpetas según tabla de arriba
3. Renombrar todos los archivos `.txt` a `.yaml` dentro de `experiment_configs/`
4. Renombrar carpeta: `configs/` → `dataset_configs/`

**Archivos de código a modificar (solo rutas de argumento):**

```bash
# run_ea_1.sh, run_fair_mo.sh, run_fairness_baseline.sh,
# run_moqnas_1.sh, run_qnas_1.sh, run_retrain.sh
# Cambiar --config_file config_files/... → --config_file experiment_configs/...
```

```yaml
# Dentro de cada experiment_configs/**/*.yaml
# Cambiar la referencia interna:
config_path_dataset: configs/cifar10.yaml
# → 
config_path_dataset: dataset_configs/cifar10.yaml
```

**Verificación:**
```bash
python run_all_evolution.py --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_rename --data_path /path/to/data \
  --dataset cifar10 --config_path_dataset dataset_configs/cifar10.yaml \
  --algo nsga2 --num_generations 1 --population_size 2 \
  --log_level INFO
```

---

### Paso 2 — Crear `settings.py` y eliminar paths hardcodeados

**Riesgo:** muy bajo | **Tiempo estimado:** 45 min

**Crear `settings.py` en la raíz:**

```python
# settings.py
import os

PROJECT_ROOT         = os.path.dirname(os.path.abspath(__file__))
DATASET_CONFIGS_DIR  = os.path.join(PROJECT_ROOT, "dataset_configs")
CFG_OBJ_PATH         = os.path.join(DATASET_CONFIGS_DIR, "cfg_obj.json")

TRAIN_TIMEOUT        = 5400  # segundos; timeout máximo por individuo en entrenamiento
```

**Modificar `algorithms/ga/nsga2.py:48`:**

```python
# Antes:
with open("configs/cfg_obj.json", "r") as f:
    self.objectives_info = json.load(f)

# Después:
from settings import CFG_OBJ_PATH
with open(CFG_OBJ_PATH, "r") as f:
    self.objectives_info = json.load(f)
```

**Modificar `algorithms/qnas/moqnas.py:221`:** mismo cambio exacto.

**Modificar `core/cnn/trainer.py:27`:**

```python
# Antes:
TRAIN_TIMEOUT = 5400

# Después:
from settings import TRAIN_TIMEOUT
```

**Verificación:**
```bash
python -c "from settings import CFG_OBJ_PATH, TRAIN_TIMEOUT; print(CFG_OBJ_PATH)"
# debe imprimir la ruta absoluta correcta

python -c "from algorithms.ga.nsga2 import NSGA2"
# no debe fallar con FileNotFoundError
```

---

### Paso 3 — Dividir `utils/helpers.py` en módulos cohesivos

**Riesgo:** bajo | **Tiempo estimado:** 2 horas

**Estrategia:** crear los 5 módulos nuevos con el código movido, luego convertir `helpers.py` en una facade de re-exports. Ningún consumidor existente se rompe hasta que decidamos eliminarlo.

**3a. Crear `utils/io.py`** — mover estas funciones desde `helpers.py`:
- `load_yaml` (L39), `_deep_merge` (L55), `_atomic_write_yaml` (L64), `update_yaml_file` (L85)
- `load_pkl` (L96), `save_pkl` (L111)
- `create_info_file` (L123), `save_results_file` (L134)
- `load_retrain_results` (L159 y L474 — hay dos; mantener solo la más completa, L474)
- `load_evolved_data` (L430) — eliminar esta; la versión canónica es `core/config.py:319`
- `load_log_params_evolution` (L529)
- `load_pareto_history` (L878)
- `load_history_from_json` (L1180), `save_history_to_json` (L1192)
- `backup_cache` (L713), `load_cache` (L746)

**3b. Crear `utils/logging_utils.py`** — mover:
- `init_log` (L396)

**3c. Crear `utils/experiment.py`** — mover:
- `natural_key` (L32)
- `check_file_exists` (L145)
- `check_files` (L345)
- `delete_old_dirs` (L325)
- `delete_old_dirs_v2` (L762)
- `calculate_time` (L566)

**3d. Crear `utils/dataset.py`** — mover:
- `dataset_cache` (L636, variable global)
- `_validate_dataset_info` (L638)
- `setup_dataset_info` (L648)
- `download_dataset` (L596)

**3e. Crear `utils/visualization.py`** — mover:
- `plot_confusion_matrix` (L165)
- `agg_results` (L183)
- `test_acc_mean_std` (L176)
- `plot_training_history` (L219)
- `get_gpu_memory` (L700)
- `compute_hypervolume_mixed` standalone (L852)
- `plot_hypervolume_over_epochs` (L887)
- `_get_hypervolume_stats` (L918)
- `plot_hypervolume_comparison` (L979)
- `plot_pareto_evolution` (L1030)
- `load_data_for_pareto` (L1123)

**3f. Convertir `utils/helpers.py` en facade:**

```python
# utils/helpers.py — facade de compatibilidad (no eliminar hasta Paso 7)
from utils.io import (
    load_yaml, save_pkl, load_pkl, create_info_file, save_results_file,
    load_log_params_evolution, load_pareto_history, load_history_from_json,
    save_history_to_json, backup_cache, load_cache, update_yaml_file,
    load_retrain_results
)
from utils.logging_utils import init_log
from utils.experiment import (
    natural_key, check_file_exists, check_files,
    delete_old_dirs, delete_old_dirs_v2, calculate_time
)
from utils.dataset import download_dataset, setup_dataset_info
from utils.visualization import (
    plot_confusion_matrix, agg_results, test_acc_mean_std,
    plot_training_history, get_gpu_memory, compute_hypervolume_mixed,
    plot_hypervolume_over_epochs, plot_hypervolume_comparison,
    plot_pareto_evolution, load_data_for_pareto
)

# Nota: load_evolved_data standalone eliminada; usar core.config.ConfigParameters.load_evolved_data
```

**Verificación:**
```bash
python -c "from utils.helpers import init_log, load_yaml, download_dataset, delete_old_dirs_v2"
python -c "from utils.io import load_yaml; from utils.logging_utils import init_log"
python -c "from utils.experiment import check_files, calculate_time"
```

---

### Paso 4 — Eliminar efectos secundarios en tiempo de import

**Riesgo:** bajo | **Tiempo estimado:** 30 min

**Modificar `core/cnn/master.py`** — eliminar líneas 23-29 y reemplazar:

```python
# Eliminar este bloque del nivel de módulo:
# project_root = os.getcwd()
# log_directory = os.path.join(project_root, 'logs')
# if not os.path.exists(log_directory):
#     os.makedirs(log_directory)
# log_file = os.path.join(log_directory, 'master.log')
# LOGGER = init_log("INFO", name=__name__, file_path=log_file)

# Reemplazar por (sin side-effects):
import logging
LOGGER = logging.getLogger(__name__)
```

**Modificar `core/cnn/trainer.py`** — mismo cambio en líneas 29-32. El logger concreto de cada instancia se configura dentro de `BaseTrainer.__init__` cuando ya se conoce el `experiment_path`.

**Verificación:**
```bash
# El siguiente comando NO debe crear el directorio logs/ ni ningún archivo
python -c "
import os, shutil
from core.cnn import master, trainer
assert not os.path.exists('logs'), 'logs/ fue creado al importar — fallo'
print('OK: sin side-effects en import')
"
```

---

### Paso 5 — Extraer operadores Pareto a `algorithms/pareto/`

**Riesgo:** medio | **Tiempo estimado:** 3 horas

**5a. Crear el paquete `algorithms/pareto/`:**

```python
# algorithms/pareto/__init__.py
from .dominance      import dominates, fast_nondominated_sort
from .diversity      import crowding_distance
from .hypervolume    import compute_hypervolume_mixed
from .reference_dirs import simplex_lattice, build_reference_directions, to_minimization
```

**5b. Crear `algorithms/pareto/dominance.py`:**

Usar como base la versión de `nsga2.py:249` (es la más correcta — aplica `objective_senses` dinámico y convierte todo a minimización antes de comparar):

```python
# algorithms/pareto/dominance.py
import numpy as np
from typing import List

def dominates(a, b, objective_senses: list) -> bool:
    obj_a = np.array(a, copy=True, dtype=float)
    obj_b = np.array(b, copy=True, dtype=float)
    for i, sense in enumerate(objective_senses):
        if sense == 'max':
            obj_a[i] = -obj_a[i]
            obj_b[i] = -obj_b[i]
    return bool(np.all(obj_a <= obj_b) and np.any(obj_a < obj_b))

def fast_nondominated_sort(fits: np.ndarray, objective_senses: list) -> List[List[int]]:
    # copiar implementación de nsga2.py:265 parametrizando objective_senses
    ...
```

**5c. Crear `algorithms/pareto/diversity.py`:**

```python
# algorithms/pareto/diversity.py
import numpy as np

def crowding_distance(fits: np.ndarray, front: list) -> np.ndarray:
    # copiar implementación de nsga2.py:300
    ...
```

**5d. Crear `algorithms/pareto/hypervolume.py`:**

```python
# algorithms/pareto/hypervolume.py
import numpy as np
from pymoo.indicators.hv import Hypervolume

def compute_hypervolume_mixed(front_raw: np.ndarray, objective_senses: list,
                               ref_point=None) -> float:
    # copiar implementación de nsga2.py:223, parametrizando objective_senses
    ...
```

**5e. Crear `algorithms/pareto/reference_dirs.py`:**

```python
# algorithms/pareto/reference_dirs.py
import numpy as np

def simplex_lattice(M: int, p: int) -> np.ndarray:
    # copiar de nsga3.py:154 (versión con más documentación)
    ...

def build_reference_directions(M: int, pop_size: int, divisions=None) -> np.ndarray:
    # copiar de nsga3.py:137
    ...

def to_minimization(fits: np.ndarray, objective_senses: list) -> np.ndarray:
    # copiar de nsga3.py:124
    ...
```

**5f. Actualizar cada algoritmo** — orden recomendado (uno por uno, verificando entre cada cambio):

1. **`algorithms/ga/nsga2.py`** — eliminar métodos `dominates`, `fast_nondominated_sort`, `crowding_distance`, `compute_hypervolume_mixed`; agregar imports desde `algorithms.pareto`
2. **`algorithms/qnas/moqnas.py`** — mismo cambio
3. **`algorithms/ga/nsga3.py`** — eliminar `_simplex_lattice`, `_build_reference_directions`, `_to_minimization`; usar `algorithms.pareto.reference_dirs`
4. **`algorithms/ga/moead.py`** — mismo cambio

**Verificación por cada algoritmo:**
```bash
# Después de migrar nsga2.py:
python run_all_evolution.py --algo nsga2 --num_generations 2 --population_size 4 \
  --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_nsga2 --data_path /path/data \
  --dataset cifar10 --config_path_dataset dataset_configs/cifar10.yaml \
  --log_level INFO
# Verificar que el frente Pareto se genera sin errores
```

---

### Paso 6 — Romper el ciclo `core ↔ algorithms`

**Riesgo:** alto | **Tiempo estimado:** 2 horas

**Hacer primero este diagnóstico:**

```bash
# Verificar si QChromosome se usa realmente en core/config.py
grep -n "QChromosome" core/config.py

# Verificar si base_ga.py usa config o evaluation de verdad
grep -n "cfg\.\|evaluation\." algorithms/ga/base_ga.py
```

**Si `QChromosomeNetwork` y `QChromosomeParams` no aparecen en ningún método** (solo en el import de la línea 15), simplemente eliminar esa línea de `core/config.py`.

**Para `algorithms/ga/base_ga.py`:** verificar si los imports `from core import evaluation` y `from core import config as cfg` se usan en el código del archivo. Si solo son imports sin uso, eliminarlos. Si se usan, mover esa lógica al script de orquestación (`run_all_evolution.py`) donde el wiring es explícito y correcto.

**Objetivo final:**
- `core/` no debe importar nada de `algorithms/`
- `algorithms/` no debe importar de `core.config` ni `core.evaluation` (puede importar de `utils/` y de `algorithms/pareto/`)

**Verificación:**
```bash
python -c "from algorithms.qnas import moqnas" 2>&1 | grep -i "core"
# No debe aparecer ningún módulo de core en la traza de import

python -c "from core import config" 2>&1 | grep -i "algorithms"
# No debe aparecer ningún módulo de algorithms en la traza de import
```

---

### Paso 7 — Limpieza final

**Riesgo:** muy bajo | **Tiempo estimado:** 1 hora

**7a. Eliminar `load_evolved_data` standalone de `utils/io.py`**

La versión canónica es `core/config.py:319` (método de clase). La función en `utils/helpers.py:430` es una versión anterior incompleta (no captura `backbone_name` ni `backbone_percentage`). Verificar que nadie la importa directamente:

```bash
grep -rn "from utils.helpers import.*load_evolved_data\|from utils.io import.*load_evolved_data" .
```

Si no hay referencias directas fuera de `core/config.py`, eliminarla de `utils/io.py` y de la facade.

**7b. Consolidar `_Wrap` en `dataset_utils/factory.py`**

Mover la clase `_Wrap` (actualmente definida en líneas 110 y 215) a nivel de módulo, antes de `build_datasets`:

```python
# dataset_utils/factory.py — antes de build_datasets()
class _TransformWrapper(Dataset):
    """Inyecta un transform en un Subset ya construido."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, i):
        x, y = self.subset[i]
        return (self.transform(x) if self.transform else x), y
```

Reemplazar ambas ocurrencias de `_Wrap` en la función por `_TransformWrapper`.

**7c. Convertir `utils/helpers.py` en advertencia de deprecación** (opcional, solo si todos los imports ya migraron):

```python
# utils/helpers.py
import warnings
warnings.warn(
    "utils.helpers está deprecado. "
    "Importa directamente de utils.io, utils.logging_utils, "
    "utils.experiment, utils.dataset o utils.visualization.",
    DeprecationWarning, stacklevel=2
)
# mantener re-exports para no romper notebooks legacy
from utils.io import *
...
```

**7d. Mover/archivar `old_files/`**

Verificar que no hay referencias desde código activo:
```bash
grep -rn "old_files" run_all_evolution.py retrain_model.py retrain_parallel.py \
  algorithms/ core/ dataset_utils/ utils/
```
Si no hay referencias: mover a una rama `git` llamada `legacy/old_files` o eliminar.

---

## Tabla resumen de cambios

| Paso | Archivos creados / renombrados | Archivos modificados | Riesgo |
|---|---|---|---|
| 1 | `experiment_configs/` (renombrar), `dataset_configs/` (renombrar) | 6 scripts `.sh`, `README.md`, todos los `.txt` dentro | Muy bajo |
| 2 | `settings.py` | `nsga2.py:48`, `moqnas.py:221`, `trainer.py:27` | Muy bajo |
| 3 | `utils/io.py`, `utils/logging_utils.py`, `utils/experiment.py`, `utils/dataset.py`, `utils/visualization.py` | `utils/helpers.py` (convertir a facade) | Bajo |
| 4 | — | `core/cnn/master.py:23-29`, `core/cnn/trainer.py:29-32` | Bajo |
| 5 | `algorithms/pareto/` (4 módulos) | `nsga2.py`, `moqnas.py`, `nsga3.py`, `moead.py` | Medio |
| 6 | — | `core/config.py:15`, `algorithms/ga/base_ga.py:7-8` | Alto |
| 7 | — | `dataset_utils/factory.py`, `utils/helpers.py` (deprecar), eliminar `old_files/` | Muy bajo |

---

## Notas para futuras sesiones

- **No hay tests automáticos.** Toda verificación es mediante ejecución de evolución corta (`--num_generations 1 --population_size 2`) o imports directos desde Python.
- **La facade `utils/helpers.py` es el mecanismo de seguridad.** Mientras exista, ningún consumidor se rompe. No eliminarla hasta que todos los módulos hayan migrado sus imports.
- **Orden crítico:** el Paso 5 (pareto) y el Paso 6 (ciclo core↔algorithms) deben hacerse en estado limpio, con git commit previo como punto de rollback.
- **El ciclo core↔algorithms (Paso 6)** puede requerir investigación adicional: hacer `grep` de uso real de `QChromosomeNetwork` y `cfg.*` en `base_ga.py` antes de modificar. Si hay uso real, la solución cambia.
- **`config_files/config_files_cifar/config_probs.ipynb`** — hay un notebook en la carpeta de configs. Después del renombrado del Paso 1, verificar que este notebook sigue accediendo a las rutas correctas.
