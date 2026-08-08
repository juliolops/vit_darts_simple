# Mapa de ruta de refactorización por etapas atómicas, MoQ-NAS

> Basado en `REFACTOR_GUIDE.md` (2026-05-04). Reordena y subdivide los 7 pasos originales en 26 etapas atómicas, cada una con red de seguridad propia (pre-verificación, post-verificación, métrica de éxito).

## Principios operativos

1. **Una etapa, un commit.** Si la post-verificación falla, se hace `git restore .` y se reanaliza, nunca se acumulan cambios.
2. **Red de seguridad antes de tocar nada.** La Etapa 0 establece la línea base contra la cual se compararán todas las verificaciones posteriores.
3. **Verificación numérica explícita en operadores MOEA.** Antes de borrar cualquier duplicado de `dominates`, `fast_nondominated_sort`, `crowding_distance`, `compute_hypervolume_mixed`, `_simplex_lattice`, `_build_reference_directions` o `_to_minimization`, se ejecuta un script de paridad que compara la salida del módulo nuevo y la del duplicado existente con los mismos inputs.
4. **Reordenamiento sobre `REFACTOR_GUIDE.md`.** Se ejecuta el bloque de `settings.py` (Paso 2) antes del bloque de renombrado de carpetas (Paso 1), para no tocar `settings.py` dos veces.

---

## ETAPA 0, línea base

**Objetivo atómico.** Establecer un estado de referencia ejecutable y reproducible contra el que comparar todas las post-verificaciones posteriores. Sin esta etapa, "no rompí nada" es una afirmación vacía.

**Script de pre-verificación.**
```bash
git status                                          # debe estar limpio
git log -1 --oneline                                # registrar SHA base
git checkout -b refactor/update-2026-staged
```

**Instrucciones de refactorización.** No se modifica código. Se generan tres artefactos de baseline en una carpeta `.refactor_baseline/` (ignorada por git):

1. Mapa de imports actuales:
   ```bash
   mkdir -p .refactor_baseline
   python -c "
   import ast, pathlib, json
   imports = {}
   for p in pathlib.Path('.').rglob('*.py'):
       if 'old_files' in p.parts or '.refactor_baseline' in p.parts: continue
       try:
           tree = ast.parse(p.read_text())
           imps = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
           imports[str(p)] = sorted(set(imps))
       except Exception: pass
   pathlib.Path('.refactor_baseline/imports.json').write_text(json.dumps(imports, indent=2))
   "
   ```

2. Smoke run de evolución corta (1 generación, población 2) con `nsga2` sobre `cifar10`, guardando stdout, frente Pareto final y hipervolumen:
   ```bash
   python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
     --config_file config_files/config_files_cifar/config0.txt \
     --experiment_path .refactor_baseline/baseline_nsga2 \
     --data_path <ruta_datos> --dataset cifar10 \
     --config_path_dataset configs/cifar10.yaml --log_level INFO \
     2>&1 | tee .refactor_baseline/baseline_nsga2.log
   ```

3. Repetir el smoke run para `moqnas` (Paso 5 lo necesita):
   ```bash
   python run_all_evolution.py --algo moqnas --num_generations 1 --population_size 2 \
     --config_file config_files/config_files_cifar_mo/config0.txt \
     --experiment_path .refactor_baseline/baseline_moqnas \
     --data_path <ruta_datos> --dataset cifar10 \
     --config_path_dataset configs/cifar10.yaml --log_level INFO \
     2>&1 | tee .refactor_baseline/baseline_moqnas.log
   ```

**Script de post-verificación.**
```bash
test -f .refactor_baseline/imports.json && \
test -f .refactor_baseline/baseline_nsga2.log && \
test -f .refactor_baseline/baseline_moqnas.log && \
echo "OK baseline"
echo ".refactor_baseline/" >> .gitignore
```

**Métrica de éxito.** Los dos smoke runs terminan sin excepciones y producen un frente Pareto con al menos 1 individuo. El hipervolumen final queda registrado en los `.log` para referencia futura. `git add .gitignore && git commit -m "refactor(0): baseline runs and import map"`.

---

## Bloque A, settings y rutas

### ETAPA A.1, crear `settings.py` apuntando a rutas antiguas

**Objetivo atómico.** Centralizar `CFG_OBJ_PATH` y `TRAIN_TIMEOUT` en un único módulo, sin tocar ningún consumidor aún. Crear solo el contenedor.

**Script de pre-verificación.**
```bash
test ! -f settings.py && echo "OK, settings.py no existe aún"
python -c "import os; assert os.path.isfile('configs/cfg_obj.json')"
grep -n "TRAIN_TIMEOUT" core/cnn/trainer.py
```

**Instrucciones de refactorización.** Crear `settings.py` en la raíz con `PROJECT_ROOT`, `DATASET_CONFIGS_DIR` (apuntando a `configs/`, ruta antigua, intencionalmente), `CFG_OBJ_PATH` y `TRAIN_TIMEOUT = 5400`. Documentar el comentario `# se actualiza en Etapa A.5 tras renombrar configs/`.

**Script de post-verificación.**
```bash
python -c "from settings import CFG_OBJ_PATH, TRAIN_TIMEOUT; import os; assert os.path.isfile(CFG_OBJ_PATH), CFG_OBJ_PATH; assert TRAIN_TIMEOUT == 5400; print('OK')"
# Asegurar que la importación no creó la carpeta logs/
test ! -d logs && echo "OK, no side-effects"
```

**Métrica de éxito.** Salida `OK` en ambos comandos. Ningún archivo `.py` existente fue modificado. `git commit -m "refactor(A.1): add settings.py"`.

---

### ETAPA A.2, migrar lectura de `cfg_obj.json` en `nsga2.py`

**Objetivo atómico.** Reemplazar la ruta relativa `"configs/cfg_obj.json"` en `algorithms/ga/nsga2.py:48` por `CFG_OBJ_PATH` de `settings`. Cambio quirúrgico de 2 líneas (import y ruta).

**Script de pre-verificación.**
```bash
grep -n "configs/cfg_obj.json" algorithms/ga/nsga2.py
python -c "from algorithms.ga.nsga2 import NSGA2; print('import OK')"
```

**Instrucciones de refactorización.** En `algorithms/ga/nsga2.py`, añadir `from settings import CFG_OBJ_PATH` (junto a los imports existentes) y reemplazar `open("configs/cfg_obj.json", "r")` por `open(CFG_OBJ_PATH, "r")`. Nada más.

**Script de post-verificación.**
```bash
grep -n "configs/cfg_obj.json" algorithms/ga/nsga2.py && echo "FAIL, sigue hardcodeado" || echo "OK, sin ruta hardcodeada"
python -c "from algorithms.ga.nsga2 import NSGA2; print('OK')"
# Smoke run corto
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --config_file config_files/config_files_cifar/config0.txt \
  --experiment_path /tmp/test_A2 --data_path <ruta_datos> --dataset cifar10 \
  --config_path_dataset configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** El smoke run de `nsga2` termina sin `FileNotFoundError` y produce un frente Pareto con el mismo número de individuos que `.refactor_baseline/baseline_nsga2.log` (igual semilla, igual configuración). `git commit -m "refactor(A.2): nsga2 reads CFG_OBJ_PATH from settings"`.

---

### ETAPA A.3, migrar lectura de `cfg_obj.json` en `moqnas.py`

**Objetivo atómico.** Aplicar el mismo cambio de A.2 en `algorithms/qnas/moqnas.py:221`.

**Script de pre-verificación.**
```bash
grep -n "configs/cfg_obj.json" algorithms/qnas/moqnas.py
```

**Instrucciones de refactorización.** Idéntico a A.2, pero en `algorithms/qnas/moqnas.py`.

**Script de post-verificación.**
```bash
grep -n "configs/cfg_obj.json" algorithms/qnas/moqnas.py && echo "FAIL" || echo "OK"
python -c "from algorithms.qnas.moqnas import MoQNAS; print('OK')"
python run_all_evolution.py --algo moqnas --num_generations 1 --population_size 2 \
  --config_file config_files/config_files_cifar_mo/config0.txt \
  --experiment_path /tmp/test_A3 --data_path <ruta_datos> --dataset cifar10 \
  --config_path_dataset configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** Smoke run de `moqnas` exitoso, frente Pareto con misma cardinalidad que baseline. `git commit -m "refactor(A.3): moqnas reads CFG_OBJ_PATH from settings"`.

---

### ETAPA A.4, migrar `TRAIN_TIMEOUT` en `trainer.py`

**Objetivo atómico.** Eliminar el literal `TRAIN_TIMEOUT = 5400` de `core/cnn/trainer.py:27` y leerlo desde `settings`.

**Script de pre-verificación.**
```bash
grep -n "TRAIN_TIMEOUT" core/cnn/trainer.py
python -c "from core.cnn.trainer import BaseTrainer; print('import OK')"
```

**Instrucciones de refactorización.** En `core/cnn/trainer.py`, reemplazar la línea `TRAIN_TIMEOUT = 5400` por `from settings import TRAIN_TIMEOUT`. No tocar el uso interno.

**Script de post-verificación.**
```bash
python -c "from core.cnn import trainer; assert trainer.TRAIN_TIMEOUT == 5400; print('OK')"
```

**Métrica de éxito.** `trainer.TRAIN_TIMEOUT` accesible y con valor 5400. `git commit -m "refactor(A.4): trainer reads TRAIN_TIMEOUT from settings"`.

---

### ETAPA A.5, renombrar `configs/` a `dataset_configs/`

**Objetivo atómico.** Renombrar una sola carpeta y actualizar las 3 ubicaciones que la referencian (settings.py, las referencias `config_path_dataset:` dentro de los `.txt` de `config_files/`, y los 6 scripts `.sh` si los usan).

**Script de pre-verificación.**
```bash
grep -rln "configs/cifar\|configs/medmnist\|configs/cfg_obj.json" \
  config_files/ algorithms/ core/ utils/ dataset_utils/ scripts/ *.py *.sh > /tmp/refs_pre.txt
cat /tmp/refs_pre.txt
```

**Instrucciones de refactorización.**
1. `git mv configs dataset_configs`
2. Actualizar `settings.py`: `DATASET_CONFIGS_DIR = os.path.join(PROJECT_ROOT, "dataset_configs")`
3. Buscar y reemplazar todas las apariciones de `configs/` a `dataset_configs/` listadas en `/tmp/refs_pre.txt`, exclusivamente las que corresponden a esta carpeta (cuidado con `config_files/` que también empieza por `config`).

**Script de post-verificación.**
```bash
test -d dataset_configs && test ! -d configs && echo "OK rename"
grep -rln "configs/cifar\|configs/medmnist\|configs/cfg_obj.json" . --include="*.py" --include="*.txt" --include="*.sh" && echo "FAIL, quedan refs" || echo "OK, sin refs viejas"
python -c "from settings import CFG_OBJ_PATH; import os; assert os.path.isfile(CFG_OBJ_PATH); print(CFG_OBJ_PATH)"
# Smoke run completo
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --config_file config_files/config_files_cifar/config0.txt \
  --experiment_path /tmp/test_A5 --data_path <ruta_datos> --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** Smoke run exitoso con la ruta nueva como argumento. `git commit -m "refactor(A.5): rename configs/ to dataset_configs/"`.

---

### ETAPA A.6, renombrar `config_files/` a `experiment_configs/` y subcarpetas

**Objetivo atómico.** Renombrar la carpeta raíz y sus 6 subcarpetas (`config_files_cifar` a `cifar`, etc.). No cambiar extensiones todavía.

**Script de pre-verificación.**
```bash
ls config_files/
grep -rln "config_files" *.sh scripts/ algorithms/ core/ utils/ dataset_utils/ *.py > /tmp/refs_cf_pre.txt
cat /tmp/refs_cf_pre.txt
```

**Instrucciones de refactorización.** Usar `git mv` para renombrar carpeta raíz y las 6 subcarpetas según la tabla del `REFACTOR_GUIDE.md`. No tocar archivos individuales aún.

**Script de post-verificación.**
```bash
test -d experiment_configs && test ! -d config_files && echo "OK"
ls experiment_configs/
# Smoke run apuntando a la nueva ruta
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --config_file experiment_configs/cifar/config0.txt \
  --experiment_path /tmp/test_A6 --data_path <ruta_datos> --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** Smoke run exitoso pasando `experiment_configs/cifar/config0.txt`. `git commit -m "refactor(A.6): rename config_files/ to experiment_configs/"`.

---

### ETAPA A.7, renombrar extensiones `.txt` a `.yaml`

**Objetivo atómico.** Renombrar todos los `.txt` dentro de `experiment_configs/**` a `.yaml`. Solo cambio de extensión; ninguna lectura de YAML cambia porque `core/config.py` usa `yaml.safe_load()`, agnóstico a la extensión.

**Script de pre-verificación.**
```bash
find experiment_configs -name "*.txt" | wc -l   # registrar cuántos archivos hay
find experiment_configs -name "*.txt" > /tmp/txt_files.txt
```

**Instrucciones de refactorización.**
```bash
find experiment_configs -name "*.txt" -exec bash -c 'git mv "$1" "${1%.txt}.yaml"' _ {} \;
```

**Script de post-verificación.**
```bash
find experiment_configs -name "*.txt" | wc -l   # debe ser 0
find experiment_configs -name "*.yaml" | wc -l  # debe coincidir con el conteo previo
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_A7 --data_path <ruta_datos> --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** Cero `.txt` restantes, mismo conteo de `.yaml`, smoke run exitoso con `.yaml`. `git commit -m "refactor(A.7): rename experiment configs .txt to .yaml"`.

---

### ETAPA A.8, actualizar los 6 scripts `.sh`

**Objetivo atómico.** Actualizar las rutas hardcodeadas dentro de `run_ea_1.sh`, `run_fair_mo.sh`, `run_fairness_baseline.sh`, `run_moqnas_1.sh`, `run_qnas_1.sh`, `run_retrain.sh` para usar `experiment_configs/.../config*.yaml`.

**Script de pre-verificación.**
```bash
grep -n "config_files\|\.txt" *.sh
```

**Instrucciones de refactorización.** Buscar y reemplazar dentro de cada `.sh` las referencias a `config_files/config_files_*` por `experiment_configs/*` y las extensiones `.txt` por `.yaml`.

**Script de post-verificación.**
```bash
grep -n "config_files\|\.txt" *.sh && echo "FAIL" || echo "OK"
# Validar que el bash parsea cada script
for f in run_*.sh; do bash -n "$f" && echo "OK $f"; done
```

**Métrica de éxito.** Sin referencias residuales, todos los `.sh` pasan `bash -n`. `git commit -m "refactor(A.8): update shell scripts to new config paths"`.

---

## Bloque B, eliminar side-effects en import

### ETAPA B.1, sanear `core/cnn/master.py`

**Objetivo atómico.** Eliminar el bloque de líneas 23-29 que crea `logs/` al hacer `import master`.

**Script de pre-verificación.**
```bash
rm -rf /tmp/B1_test && mkdir /tmp/B1_test && cd /tmp/B1_test && \
  python -c "import sys; sys.path.insert(0, '<ruta_proyecto>'); from core.cnn import master" && \
  ls -la /tmp/B1_test
# Si aparece logs/ aquí, el side-effect está confirmado pre-cambio.
```

**Instrucciones de refactorización.** En `core/cnn/master.py`, reemplazar las líneas 23-29 por:
```python
import logging
LOGGER = logging.getLogger(__name__)
```
No tocar `BaseTrainer` ni el resto del archivo. El logger concreto se configura aguas abajo cuando se conoce el `experiment_path`.

**Script de post-verificación.**
```bash
rm -rf /tmp/B1_test && mkdir /tmp/B1_test && cd /tmp/B1_test && \
  python -c "import sys; sys.path.insert(0, '<ruta_proyecto>'); from core.cnn import master" && \
  test ! -d logs && echo "OK, no side-effect"
```

**Métrica de éxito.** Importar `master` desde un CWD vacío no crea ningún archivo ni directorio. `git commit -m "refactor(B.1): remove import-time side-effects in master.py"`.

---

### ETAPA B.2, sanear `core/cnn/trainer.py`

**Objetivo atómico.** Idéntico a B.1, en `core/cnn/trainer.py` líneas 29-32.

**Script de pre-verificación.**
```bash
rm -rf /tmp/B2_test && mkdir /tmp/B2_test && cd /tmp/B2_test && \
  python -c "import sys; sys.path.insert(0, '<ruta_proyecto>'); from core.cnn import trainer"; \
  ls -la /tmp/B2_test
```

**Instrucciones de refactorización.** Reemplazar el bloque de inicialización de logger por `import logging; LOGGER = logging.getLogger(__name__)`.

**Script de post-verificación.**
```bash
rm -rf /tmp/B2_test && mkdir /tmp/B2_test && cd /tmp/B2_test && \
  python -c "import sys; sys.path.insert(0, '<ruta_proyecto>'); from core.cnn import trainer" && \
  test ! -d logs && echo "OK"
# Smoke run de entrenamiento real para confirmar que LOGGER sigue funcionando
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_B2 --data_path <ruta_datos> --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** Importar `trainer` no crea side-effects y un smoke run completo termina sin error de logger. `git commit -m "refactor(B.2): remove import-time side-effects in trainer.py"`.

---

## Bloque C, dividir `utils/helpers.py`

Estrategia: 5 sub-etapas crean los módulos nuevos sin tocar `helpers.py`. La sexta sub-etapa convierte `helpers.py` en facade. Esto preserva todos los `from utils.helpers import X` existentes hasta el final.

### ETAPA C.1, crear `utils/io.py`

**Objetivo atómico.** Mover las funciones de I/O (lista detallada en `REFACTOR_GUIDE.md` sección 3a) a un módulo nuevo, conservando exactamente las firmas y cuerpos. **Exclusión:** no mover `load_evolved_data` standalone (se elimina en F.2).

**Script de pre-verificación.**
```bash
python -c "from utils.helpers import load_yaml, save_pkl, load_pkl, update_yaml_file, backup_cache, load_cache, load_log_params_evolution, load_pareto_history, load_history_from_json, save_history_to_json, load_retrain_results, create_info_file, save_results_file; print('OK pre')"
```

**Instrucciones de refactorización.** Crear `utils/io.py`. Copiar las funciones listadas desde `utils/helpers.py` con sus imports propios (yaml, pickle, json, pathlib, os). Por ahora `helpers.py` queda intacto, ambos módulos contendrán las mismas funciones temporalmente.

**Script de post-verificación.**
```bash
python -c "from utils.io import load_yaml, save_pkl, load_pkl, update_yaml_file, backup_cache, load_cache, load_log_params_evolution, load_pareto_history, load_history_from_json, save_history_to_json, load_retrain_results, create_info_file, save_results_file; print('OK post')"
# Verificar paridad sobre load_yaml con un archivo conocido
python -c "
from utils.helpers import load_yaml as old
from utils.io import load_yaml as new
r1 = old('dataset_configs/cifar10.yaml')
r2 = new('dataset_configs/cifar10.yaml')
assert r1 == r2, 'divergencia en load_yaml'
print('OK paridad')
"
```

**Métrica de éxito.** Ambos imports funcionan, paridad confirmada sobre `load_yaml`. `git commit -m "refactor(C.1): add utils/io.py (helpers.py untouched)"`.

---

### ETAPA C.2, crear `utils/logging_utils.py`

**Objetivo atómico.** Mover `init_log` (línea 396 de `helpers.py`) a `utils/logging_utils.py`.

**Script de pre-verificación.**
```bash
python -c "from utils.helpers import init_log; print(init_log)"
```

**Instrucciones de refactorización.** Crear `utils/logging_utils.py` con `init_log` y sus imports (`logging`, `os`).

**Script de post-verificación.**
```bash
python -c "from utils.logging_utils import init_log; print('OK')"
# Verificar que no crea side-effects al importar
rm -rf /tmp/C2_test && mkdir /tmp/C2_test && cd /tmp/C2_test && \
  python -c "import sys; sys.path.insert(0, '<ruta_proyecto>'); from utils.logging_utils import init_log" && \
  test ! -d logs && echo "OK sin side-effects"
```

**Métrica de éxito.** Import exitoso sin crear `logs/`. `git commit -m "refactor(C.2): add utils/logging_utils.py"`.

---

### ETAPA C.3, crear `utils/experiment.py`

**Objetivo atómico.** Mover `natural_key`, `check_file_exists`, `check_files`, `delete_old_dirs`, `delete_old_dirs_v2`, `calculate_time`.

**Script de pre-verificación.**
```bash
python -c "from utils.helpers import natural_key, check_file_exists, check_files, delete_old_dirs, delete_old_dirs_v2, calculate_time; print('OK')"
```

**Instrucciones de refactorización.** Crear `utils/experiment.py` con las funciones listadas y sus dependencias (re, os, shutil, datetime, time).

**Script de post-verificación.**
```bash
python -c "from utils.experiment import natural_key, check_file_exists, check_files, delete_old_dirs, delete_old_dirs_v2, calculate_time; print('OK')"
python -c "
from utils.helpers import natural_key as old
from utils.experiment import natural_key as new
assert old('exp_10') == new('exp_10')
assert old('exp_2') == new('exp_2')
print('OK paridad')
"
```

**Métrica de éxito.** Imports y paridad sobre `natural_key`. `git commit -m "refactor(C.3): add utils/experiment.py"`.

---

### ETAPA C.4, crear `utils/dataset.py`

**Objetivo atómico.** Mover `download_dataset`, `setup_dataset_info`, `dataset_cache`, `_validate_dataset_info`.

**Script de pre-verificación.**
```bash
python -c "from utils.helpers import download_dataset, setup_dataset_info; print('OK')"
```

**Instrucciones de refactorización.** Crear `utils/dataset.py` con las funciones y sus imports (torchvision, medmnist, etc.). Mantener `dataset_cache` como variable global del módulo.

**Script de post-verificación.**
```bash
python -c "from utils.dataset import download_dataset, setup_dataset_info, dataset_cache; print('OK')"
# Verificar que no se descarga nada al importar
test ! -d /tmp/spurious_download && echo "OK"
```

**Métrica de éxito.** Imports sin side-effects de descarga. `git commit -m "refactor(C.4): add utils/dataset.py"`.

---

### ETAPA C.5, crear `utils/visualization.py`

**Objetivo atómico.** Mover todas las funciones de plotting y agregación de resultados listadas en `REFACTOR_GUIDE.md` sección 3e.

**Script de pre-verificación.**
```bash
python -c "from utils.helpers import plot_training_history, agg_results, plot_hypervolume_comparison, plot_pareto_evolution; print('OK')"
```

**Instrucciones de refactorización.** Crear `utils/visualization.py` con las funciones y sus imports (matplotlib, plotly, numpy, pandas, GPUtil). **Nota crítica:** la función standalone `compute_hypervolume_mixed` en línea 852 de `helpers.py` se mueve aquí temporalmente; será eliminada en D.6 cuando todos los algoritmos usen `algorithms.pareto.hypervolume`.

**Script de post-verificación.**
```bash
python -c "from utils.visualization import plot_training_history, agg_results, plot_hypervolume_comparison, plot_pareto_evolution, compute_hypervolume_mixed; print('OK')"
```

**Métrica de éxito.** Imports funcionales. `git commit -m "refactor(C.5): add utils/visualization.py"`.

---

### ETAPA C.6, convertir `utils/helpers.py` en facade

**Objetivo atómico.** Vaciar `helpers.py` y reemplazarlo por re-exports desde los 5 módulos nuevos. Cero consumidores deben romperse.

**Script de pre-verificación.**
```bash
# Inventario de imports actuales de helpers a lo largo del proyecto
grep -rn "from utils.helpers import\|from utils import helpers" . --include="*.py" > /tmp/helpers_consumers.txt
wc -l /tmp/helpers_consumers.txt
```

**Instrucciones de refactorización.** Reemplazar el contenido de `utils/helpers.py` por re-exports tal como muestra `REFACTOR_GUIDE.md` sección 3f. Excluir `load_evolved_data` (no estaba ahí en C.1 y se elimina formalmente en F.2).

**Script de post-verificación.**
```bash
# Verificar que cada símbolo importado por consumidores sigue accesible
python -c "
from utils.helpers import (
  load_yaml, save_pkl, load_pkl, init_log, download_dataset,
  setup_dataset_info, delete_old_dirs_v2, check_files, calculate_time,
  plot_training_history, plot_hypervolume_comparison, compute_hypervolume_mixed,
  agg_results, backup_cache, load_cache
)
print('OK facade')
"
# Smoke run completo
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_C6 --data_path <ruta_datos> --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** Smoke run exitoso, hipervolumen final dentro del 1% del baseline. `wc -l utils/helpers.py` debe ser < 30 líneas. `git commit -m "refactor(C.6): convert utils/helpers.py to facade"`.

---

## Bloque D, operadores Pareto

Estrategia: crear los módulos `algorithms/pareto/*` sin borrar duplicados (D.1 a D.4). Después de cada creación, ejecutar paridad numérica contra el duplicado existente (D.5). Solo entonces migrar cada algoritmo uno por uno (D.6 a D.9), borrando duplicados a medida que se migran.

### ETAPA D.1, crear `algorithms/pareto/dominance.py`

**Objetivo atómico.** Crear el módulo `dominance.py` con `dominates` y `fast_nondominated_sort`, copiando desde `nsga2.py:249` y `nsga2.py:265` (versiones canónicas según el diagnóstico).

**Script de pre-verificación.**
```bash
test ! -d algorithms/pareto && echo "OK, no existe aún"
grep -n "def dominates\|def fast_nondominated_sort" algorithms/ga/nsga2.py algorithms/qnas/moqnas.py
```

**Instrucciones de refactorización.** Crear `algorithms/pareto/__init__.py` (vacío por ahora) y `algorithms/pareto/dominance.py` con las dos funciones, parametrizadas por `objective_senses: list`. No tocar `nsga2.py` ni `moqnas.py`.

**Script de post-verificación.**
```bash
python -c "from algorithms.pareto.dominance import dominates, fast_nondominated_sort; print('OK import')"
# Paridad contra duplicado de nsga2
python -c "
import numpy as np
from algorithms.pareto.dominance import dominates as new_dom, fast_nondominated_sort as new_fns
from algorithms.ga import nsga2
np.random.seed(0)
fits = np.random.rand(10, 3)
senses = ['max', 'min', 'min']
# Comparar dominates pairwise
inst = nsga2.NSGA2.__new__(nsga2.NSGA2)
inst.objectives_info = {'objectives': [{'sense': s} for s in senses]}
for i in range(10):
  for j in range(10):
    assert new_dom(fits[i], fits[j], senses) == inst._dominates(fits[i], fits[j])
print('OK dominates paridad')
"
```

**Métrica de éxito.** Paridad bit-a-bit (Boolean) sobre 100 comparaciones aleatorias. `git commit -m "refactor(D.1): add algorithms/pareto/dominance.py"`.

---

### ETAPA D.2, crear `algorithms/pareto/diversity.py`

**Objetivo atómico.** Crear `crowding_distance` en módulo separado.

**Script de pre-verificación.**
```bash
grep -n "def crowding_distance\|def _crowding_distance" algorithms/ga/nsga2.py algorithms/qnas/moqnas.py
```

**Instrucciones de refactorización.** Copiar `crowding_distance` de `nsga2.py:300` a `algorithms/pareto/diversity.py`.

**Script de post-verificación.**
```bash
python -c "
import numpy as np
from algorithms.pareto.diversity import crowding_distance as new_cd
from algorithms.ga import nsga2
np.random.seed(1)
fits = np.random.rand(20, 3)
front = list(range(20))
inst = nsga2.NSGA2.__new__(nsga2.NSGA2)
d_new = new_cd(fits, front)
d_old = inst._crowding_distance(fits, front)
assert np.allclose(d_new, d_old, equal_nan=True), 'divergencia'
print('OK crowding paridad')
"
```

**Métrica de éxito.** `np.allclose` exitoso con 20 puntos en 3D. `git commit -m "refactor(D.2): add algorithms/pareto/diversity.py"`.

---

### ETAPA D.3, crear `algorithms/pareto/hypervolume.py`

**Objetivo atómico.** Crear `compute_hypervolume_mixed` canónico, parametrizado por `objective_senses` y opcionalmente `ref_point`.

**Script de pre-verificación.**
```bash
grep -n "def compute_hypervolume_mixed\|def _compute_hypervolume_mixed" algorithms/ga/nsga2.py algorithms/qnas/moqnas.py utils/helpers.py utils/visualization.py
```

**Instrucciones de refactorización.** Crear `algorithms/pareto/hypervolume.py` basado en la versión de `nsga2.py:223`. Importar `pymoo.indicators.hv.Hypervolume`.

**Script de post-verificación.** **Verificación clave que pediste explícitamente:**
```bash
python -c "
import numpy as np
from algorithms.pareto.hypervolume import compute_hypervolume_mixed as new_hv
from algorithms.ga import nsga2
np.random.seed(42)
# Frente con valores típicos: accuracy en [0,1] (max), size en [0,10] (min), latency en [0,100] (min)
front = np.column_stack([
    np.random.uniform(0.5, 0.95, 30),
    np.random.uniform(0.1, 8.0, 30),
    np.random.uniform(5.0, 80.0, 30),
])
senses = ['max', 'min', 'min']
inst = nsga2.NSGA2.__new__(nsga2.NSGA2)
inst.objectives_info = {'objectives': [{'sense': s} for s in senses]}
hv_new = new_hv(front, senses)
hv_old = inst._compute_hypervolume_mixed(front)
assert np.isclose(hv_new, hv_old, rtol=1e-12), f'HV divergence: new={hv_new}, old={hv_old}'
print(f'OK HV paridad: {hv_new}')
"
```

**Métrica de éxito.** `np.isclose` con `rtol=1e-12`. `git commit -m "refactor(D.3): add algorithms/pareto/hypervolume.py"`.

---

### ETAPA D.4, crear `algorithms/pareto/reference_dirs.py`

**Objetivo atómico.** Crear `simplex_lattice`, `build_reference_directions`, `to_minimization` desde `nsga3.py:124-154` (más documentada que la de `moead.py`).

**Script de pre-verificación.**
```bash
grep -n "def _simplex_lattice\|def _build_reference_directions\|def _to_minimization" algorithms/ga/nsga3.py algorithms/ga/moead.py
```

**Instrucciones de refactorización.** Copiar las tres funciones a `algorithms/pareto/reference_dirs.py` sin el guion bajo inicial (ya no son privadas).

**Script de post-verificación.**
```bash
python -c "
import numpy as np
from algorithms.pareto.reference_dirs import simplex_lattice as new_sl, to_minimization as new_tm
from algorithms.ga import nsga3, moead
sl_new = new_sl(M=3, p=12)
sl_old_nsga3 = nsga3.NSGA3._simplex_lattice(None, M=3, p=12)
sl_old_moead = moead.MOEAD._simplex_lattice(None, M=3, p=12)
assert np.allclose(sl_new, sl_old_nsga3) and np.allclose(sl_new, sl_old_moead), 'divergencia simplex'

fits = np.random.rand(15, 3)
senses = ['max', 'min', 'min']
tm_new = new_tm(fits, senses)
tm_old = nsga3.NSGA3._to_minimization(None, fits, senses)
assert np.allclose(tm_new, tm_old), 'divergencia to_minimization'
print('OK reference_dirs paridad')
"
```

**Métrica de éxito.** Paridad exacta de `simplex_lattice` entre las tres versiones (nueva, nsga3 vieja, moead vieja) y de `to_minimization`. `git commit -m "refactor(D.4): add algorithms/pareto/reference_dirs.py"`.

---

### ETAPA D.5, completar `algorithms/pareto/__init__.py`

**Objetivo atómico.** Exponer las 7 funciones del paquete `pareto` por su API pública.

**Script de pre-verificación.**
```bash
cat algorithms/pareto/__init__.py | wc -l   # debe ser 0 o casi 0
```

**Instrucciones de refactorización.** Llenar `algorithms/pareto/__init__.py` con:
```python
from .dominance      import dominates, fast_nondominated_sort
from .diversity      import crowding_distance
from .hypervolume    import compute_hypervolume_mixed
from .reference_dirs import simplex_lattice, build_reference_directions, to_minimization
```

**Script de post-verificación.**
```bash
python -c "from algorithms.pareto import (dominates, fast_nondominated_sort, crowding_distance, compute_hypervolume_mixed, simplex_lattice, build_reference_directions, to_minimization); print('OK')"
```

**Métrica de éxito.** Un import desde el paquete trae los 7 símbolos. `git commit -m "refactor(D.5): expose algorithms/pareto public API"`.

---

### ETAPA D.6, migrar `nsga2.py` y borrar duplicados

**Objetivo atómico.** En `algorithms/ga/nsga2.py`, eliminar las definiciones internas de `_dominates`, `_fast_nondominated_sort`, `_crowding_distance`, `_compute_hypervolume_mixed`, y reemplazar sus llamadas por `from algorithms.pareto import ...`.

**Script de pre-verificación.**
```bash
# Guardar el HV final del baseline para comparar
grep -i "hypervolume\|hv" .refactor_baseline/baseline_nsga2.log | tail -5
```

**Instrucciones de refactorización.**
1. Añadir `from algorithms.pareto import dominates, fast_nondominated_sort, crowding_distance, compute_hypervolume_mixed` al tope del archivo.
2. Eliminar las 4 definiciones internas.
3. Reemplazar cada `self._dominates(a, b)` por `dominates(a, b, self.objective_senses)`. Misma transformación para las otras tres funciones (extraer `objective_senses` desde `self.objectives_info` o pasarlo como atributo).

**Script de post-verificación.**
```bash
grep -n "def _dominates\|def _fast_nondominated\|def _crowding_distance\|def _compute_hypervolume" algorithms/ga/nsga2.py && echo "FAIL, quedan duplicados" || echo "OK, duplicados eliminados"
# Smoke run con MISMA semilla del baseline
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_D6 --data_path <ruta_datos> --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO \
  2>&1 | tee /tmp/test_D6.log
# Comparar HV final
diff <(grep -i "hypervolume" .refactor_baseline/baseline_nsga2.log | tail -1) \
     <(grep -i "hypervolume" /tmp/test_D6.log | tail -1)
```

**Métrica de éxito.** HV final idéntico (o con diferencia < 1e-10) al baseline registrado en Etapa 0. `git commit -m "refactor(D.6): nsga2 uses algorithms/pareto, remove duplicates"`.

---

### ETAPA D.7, migrar `moqnas.py` y borrar duplicados

**Objetivo atómico.** Misma operación que D.6 sobre `algorithms/qnas/moqnas.py`. **Verificación crítica adicional:** confirmar que la asignación de pesos `1/rank` para la élite sigue funcionando idénticamente.

**Script de pre-verificación.**
```bash
grep -n "1/rank\|1 / rank\|elite_weight\|rank_weight" algorithms/qnas/moqnas.py
# Registrar baseline de HV de moqnas
grep -i "hypervolume" .refactor_baseline/baseline_moqnas.log | tail -1
```

**Instrucciones de refactorización.** Idénticas a D.6 pero sobre `moqnas.py`. La lógica de élite ponderada por `1/rank` está fuera de los operadores Pareto; vive en el código de actualización cuántica. No se toca esa parte, solo se sustituyen las llamadas a `fast_nondominated_sort` (que produce los ranks que después alimentan `1/rank`).

**Script de post-verificación.**
```bash
grep -n "def _dominates\|def _fast_nondominated\|def _crowding_distance\|def _compute_hypervolume" algorithms/qnas/moqnas.py && echo "FAIL" || echo "OK"
python run_all_evolution.py --algo moqnas --num_generations 1 --population_size 2 \
  --config_file experiment_configs/cifar_mo/config0.yaml \
  --experiment_path /tmp/test_D7 --data_path <ruta_datos> --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO \
  2>&1 | tee /tmp/test_D7.log
# Verificación específica del flujo 1/rank: log de la primera generación debe mostrar
# los mismos pesos de élite que el baseline
diff <(grep -E "rank|elite" .refactor_baseline/baseline_moqnas.log | head -20) \
     <(grep -E "rank|elite" /tmp/test_D7.log | head -20)
```

**Métrica de éxito.** HV final del frente moqnas idéntico al baseline; los logs de rank y pesos de élite coinciden en la primera generación. `git commit -m "refactor(D.7): moqnas uses algorithms/pareto, remove duplicates"`.

---

### ETAPA D.8, migrar `nsga3.py` y borrar duplicados

**Objetivo atómico.** Eliminar `_simplex_lattice`, `_build_reference_directions`, `_to_minimization` de `algorithms/ga/nsga3.py` y usar `algorithms.pareto.reference_dirs`.

**Script de pre-verificación.**
```bash
grep -n "def _simplex_lattice\|def _build_reference_directions\|def _to_minimization" algorithms/ga/nsga3.py
# Registrar baseline de HV nsga3
python run_all_evolution.py --algo nsga3 --num_generations 1 --population_size 2 \
  --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path .refactor_baseline/baseline_nsga3 \
  --data_path <ruta_datos> --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO \
  2>&1 | tee .refactor_baseline/baseline_nsga3.log
```

**Instrucciones de refactorización.** Añadir imports desde `algorithms.pareto.reference_dirs`. Eliminar las 3 definiciones internas. Reemplazar llamadas `self._...` por las funciones libres.

**Script de post-verificación.**
```bash
grep -n "def _simplex_lattice\|def _build_reference_directions\|def _to_minimization" algorithms/ga/nsga3.py && echo "FAIL" || echo "OK"
python run_all_evolution.py --algo nsga3 --num_generations 1 --population_size 2 \
  --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_D8 --data_path <ruta_datos> --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO \
  2>&1 | tee /tmp/test_D8.log
diff <(grep -i "hypervolume" .refactor_baseline/baseline_nsga3.log | tail -1) \
     <(grep -i "hypervolume" /tmp/test_D8.log | tail -1)
```

**Métrica de éxito.** HV final idéntico al baseline nsga3. `git commit -m "refactor(D.8): nsga3 uses pareto.reference_dirs"`.

---

### ETAPA D.9, migrar `moead.py` y borrar duplicados

**Objetivo atómico.** Igual que D.8, en `algorithms/ga/moead.py`.

**Script de pre-verificación.**
```bash
grep -n "def _simplex_lattice\|def _build_reference_directions\|def _to_minimization" algorithms/ga/moead.py
python run_all_evolution.py --algo moead --num_generations 1 --population_size 2 \
  --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path .refactor_baseline/baseline_moead \
  --data_path <ruta_datos> --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO \
  2>&1 | tee .refactor_baseline/baseline_moead.log
```

**Instrucciones de refactorización.** Idénticas a D.8 sobre `moead.py`.

**Script de post-verificación.**
```bash
grep -n "def _simplex_lattice\|def _build_reference_directions\|def _to_minimization" algorithms/ga/moead.py && echo "FAIL" || echo "OK"
python run_all_evolution.py --algo moead --num_generations 1 --population_size 2 \
  --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_D9 --data_path <ruta_datos> --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO \
  2>&1 | tee /tmp/test_D9.log
diff <(grep -i "hypervolume" .refactor_baseline/baseline_moead.log | tail -1) \
     <(grep -i "hypervolume" /tmp/test_D9.log | tail -1)
```

**Métrica de éxito.** HV final idéntico al baseline moead. `git commit -m "refactor(D.9): moead uses pareto.reference_dirs"`.

---

## Bloque E, romper ciclo `core ↔ algorithms`

### ETAPA E.1, diagnóstico de uso real

**Objetivo atómico.** Decidir empíricamente si los imports cruzados son usados o son dead code. Sin escribir nada. Esta etapa no produce commit.

**Script de pre-verificación.** No aplica (etapa de solo lectura).

**Instrucciones de refactorización.** Ejecutar los comandos diagnósticos:
```bash
echo "=== ¿Se usa QChromosome en core/config.py? ==="
grep -n "QChromosomeNetwork\|QChromosomeParams" core/config.py

echo "=== ¿Se usa cfg.* o evaluation.* en base_ga.py? ==="
grep -n "cfg\.\|evaluation\." algorithms/ga/base_ga.py

echo "=== ¿Quién llama a base_ga directamente desde core? ==="
grep -rn "from algorithms.ga.base_ga\|from algorithms.ga import base_ga" core/
```

**Script de post-verificación.** Documentar los hallazgos en un archivo `.refactor_baseline/E1_diagnostico.md` con tres posibles escenarios:
- **Caso 1 (limpio):** Los imports no se usan; eliminar líneas en E.2 y E.3 será trivial.
- **Caso 2 (usado en core):** `QChromosome` se referencia en algún método; requiere extraer interface o mover lógica.
- **Caso 3 (usado en base_ga):** `cfg.*` o `evaluation.*` aparece; mover el wiring a `run_all_evolution.py`.

**Métrica de éxito.** El archivo `E1_diagnostico.md` contiene los outputs de los 3 greps y la decisión de qué escenario aplica. No hay commit.

---

### ETAPA E.2, romper `core/config.py → algorithms`

**Objetivo atómico.** Eliminar (o reubicar) el import `from algorithms.qnas.chromosome import QChromosomeNetwork, QChromosomeParams` en `core/config.py:15`. La acción concreta depende del escenario detectado en E.1.

**Script de pre-verificación.**
```bash
python -c "from core import config" 2>&1 | tee /tmp/E2_pre.log
python -c "
import sys
from core import config
mods = [m for m in sys.modules if 'algorithms' in m]
print('Módulos algorithms cargados al importar core.config:', mods)
"
```

**Instrucciones de refactorización (escenario 1 según E.1).** Eliminar la línea 15. Si `QChromosomeNetwork` aparece como string en `evolved_params`, no requiere import (es solo metadato). Si E.1 detectó uso real, la instrucción será diferente; documentar la elección en el mensaje de commit.

**Script de post-verificación.**
```bash
python -c "
import sys
from core import config
mods = [m for m in sys.modules if m.startswith('algorithms')]
assert not mods, f'core.config sigue cargando: {mods}'
print('OK, core.config no importa algorithms')
"
# Smoke run para confirmar que no rompimos nada
python run_all_evolution.py --algo moqnas --num_generations 1 --population_size 2 \
  --config_file experiment_configs/cifar_mo/config0.yaml \
  --experiment_path /tmp/test_E2 --data_path <ruta_datos> --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** `core.config` no carga ningún módulo de `algorithms` al ser importado; smoke run de `moqnas` (que sí usa `QChromosome` legítimamente) sigue funcionando. `git commit -m "refactor(E.2): break core.config -> algorithms import"`.

---

### ETAPA E.3, romper `algorithms/ga/base_ga.py → core`

**Objetivo atómico.** Eliminar (o reubicar) `from core import evaluation` y `from core import config as cfg` en `base_ga.py:7-8`.

**Script de pre-verificación.**
```bash
python -c "
import sys
from algorithms.ga import base_ga
mods = [m for m in sys.modules if m.startswith('core')]
print('Módulos core cargados al importar base_ga:', mods)
"
```

**Instrucciones de refactorización.** Según escenario E.1:
- Si no se usan: eliminar las dos líneas.
- Si se usan: mover las llamadas a `cfg.*` y `evaluation.*` a `run_all_evolution.py` y pasar las referencias al constructor de `BaseGA` (inyección de dependencias explícita).

**Script de post-verificación.**
```bash
python -c "
import sys
# Limpiar sys.modules para test limpio
for m in list(sys.modules.keys()):
    if 'algorithms' in m or 'core' in m: del sys.modules[m]
from algorithms.ga import base_ga
mods = [m for m in sys.modules if m.startswith('core')]
assert not mods, f'base_ga sigue cargando: {mods}'
print('OK, base_ga no importa core')
"
# Smoke run de los 4 algoritmos GA
for algo in nsga2 nsga3 moead; do
  python run_all_evolution.py --algo $algo --num_generations 1 --population_size 2 \
    --config_file experiment_configs/cifar/config0.yaml \
    --experiment_path /tmp/test_E3_$algo --data_path <ruta_datos> --dataset cifar10 \
    --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO
done
```

**Métrica de éxito.** Los 3 algoritmos GA terminan sin error y producen frente Pareto no vacío. `git commit -m "refactor(E.3): break algorithms.base_ga -> core import"`.

---

## Bloque F, limpieza final

### ETAPA F.1, consolidar `_Wrap` en `dataset_utils/factory.py`

**Objetivo atómico.** Mover la clase `_Wrap` (definida dos veces en líneas 110 y 215) a nivel de módulo como `_TransformWrapper`, eliminando la duplicación.

**Script de pre-verificación.**
```bash
grep -n "class _Wrap" dataset_utils/factory.py | wc -l   # debe ser 2
```

**Instrucciones de refactorización.** Definir `_TransformWrapper` a nivel de módulo, antes de `build_datasets`. Reemplazar las dos definiciones internas por referencias a `_TransformWrapper`.

**Script de post-verificación.**
```bash
grep -n "class _Wrap\|class _TransformWrapper" dataset_utils/factory.py
# Smoke run con cifar (rama torchvision) y con un dataset binario para cubrir ambas ramas
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_F1_cifar --data_path <ruta_datos> --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** Solo una definición de `_TransformWrapper` en el archivo; smoke run exitoso. `git commit -m "refactor(F.1): consolidate _Wrap into _TransformWrapper"`.

---

### ETAPA F.2, eliminar `load_evolved_data` standalone

**Objetivo atómico.** Eliminar la versión incompleta de `load_evolved_data` (la que estuvo en `helpers.py:430` y que en C.1 decidimos no portar a `utils/io.py`). Verificar que nadie la importa directamente.

**Script de pre-verificación.**
```bash
grep -rn "from utils.helpers import.*load_evolved_data\|from utils.io import.*load_evolved_data\|helpers.load_evolved_data\|io.load_evolved_data" . --include="*.py"
```

**Instrucciones de refactorización.** Si el grep no devuelve resultados, confirmar que la función ya no existe en ningún módulo `utils/*`. Si quedó en el cuerpo (no en `__all__`) de algún archivo, eliminar. Documentar que la versión canónica es `core/config.ConfigParameters.load_evolved_data`.

**Script de post-verificación.**
```bash
grep -rn "def load_evolved_data" utils/   # no debe haber resultados
grep -n "def load_evolved_data" core/config.py   # debe seguir existiendo
python -c "from core.config import ConfigParameters; print('OK')"
```

**Métrica de éxito.** Cero definiciones de `load_evolved_data` en `utils/`, una sola en `core/config.py`. `git commit -m "refactor(F.2): remove standalone load_evolved_data"`.

---

### ETAPA F.3, marcar `utils/helpers.py` como deprecated

**Objetivo atómico.** Añadir `DeprecationWarning` al inicio de la facade. Opcional, no destructivo.

**Script de pre-verificación.**
```bash
wc -l utils/helpers.py   # debe ser < 30 (post C.6)
python -W error::DeprecationWarning -c "from utils.helpers import load_yaml" 2>&1 | grep -i "deprecat" && echo "ya hay warning" || echo "sin warning aún"
```

**Instrucciones de refactorización.** Añadir al tope del archivo:
```python
import warnings
warnings.warn(
    "utils.helpers está deprecado. "
    "Importa directamente desde utils.io, utils.logging_utils, "
    "utils.experiment, utils.dataset o utils.visualization.",
    DeprecationWarning, stacklevel=2
)
```

**Script de post-verificación.**
```bash
python -W default::DeprecationWarning -c "from utils.helpers import load_yaml" 2>&1 | grep -i "deprecat" && echo "OK warning emitido"
# Verificar que el código de producción no genera DeprecationWarnings ya (significaría que C.6 quedó incompleto)
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_F3 --data_path <ruta_datos> --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO \
  2>&1 | grep -i "deprecat"
```

**Métrica de éxito.** El warning se emite al importar `helpers` desde el exterior; el smoke run no emite ningún `DeprecationWarning` (confirma que todo el código interno ya migró). `git commit -m "refactor(F.3): deprecate utils/helpers.py facade"`.

---

### ETAPA F.4, archivar `old_files/`

**Objetivo atómico.** Confirmar que `old_files/` no está referenciado desde código activo y archivarlo (rama git o eliminación).

**Script de pre-verificación.**
```bash
grep -rn "old_files" run_all_evolution.py retrain_model.py retrain_parallel.py \
  algorithms/ core/ dataset_utils/ utils/ scripts/ --include="*.py"
```

**Instrucciones de refactorización.** Si el grep no devuelve resultados:
```bash
git checkout -b legacy/old_files
git push origin legacy/old_files
git checkout refactor/update-2026-staged
git rm -r old_files/
```

**Script de post-verificación.**
```bash
test ! -d old_files && echo "OK, old_files eliminado"
git ls-remote origin legacy/old_files | grep "legacy/old_files" && echo "OK, backup en rama"
# Smoke run final
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_F4 --data_path <ruta_datos> --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** Carpeta eliminada localmente, backup empujado a rama remota `legacy/old_files`, smoke run final exitoso. `git commit -m "refactor(F.4): archive old_files/ to legacy branch"`.

---

## Resumen de etapas

| ID | Bloque | Riesgo | Commit esperado |
|---|---|---|---|
| 0 | Línea base | nulo | `baseline runs and import map` |
| A.1–A.8 | Settings y renombrados | muy bajo | 8 commits de configuración |
| B.1–B.2 | Side-effects de import | bajo | 2 commits |
| C.1–C.6 | Split de helpers | bajo | 6 commits |
| D.1–D.9 | Operadores Pareto | medio | 9 commits con verificación numérica |
| E.1–E.3 | Ciclo core/algorithms | alto | 1 diagnóstico + 2 commits |
| F.1–F.4 | Limpieza | muy bajo | 4 commits |

**Total: 32 commits atómicos.** Cada uno con su propio rollback (`git revert <sha>`) en caso de descubrir regresión semanas después.

## Notas de validación para Diego

1. **Reemplazar `<ruta_datos>` en todos los scripts** por la ruta real de tu dataset antes de ejecutar.
2. **Antes de la Etapa D**, asegúrate de que la semilla de evolución es determinista; si `numpy.random.seed` o `torch.manual_seed` no están fijados en `run_all_evolution.py`, los `diff` de hipervolumen de las etapas D.6–D.9 fallarán por ruido estocástico, no por refactor. Si ese es el caso, conviene añadir una etapa intermedia D.0 que fije semillas y re-genere los baselines.
3. **La Etapa E.1 es solo lectura.** Si el diagnóstico revela el escenario 2 o 3 (uso real, no dead code), el plan de E.2 y E.3 cambia y conviene re-validar el roadmap antes de continuar.
4. **No hay tests automáticos hoy.** Cada smoke run consume tiempo de GPU. Estimación total de runs: ~30 ejecuciones de 1 generación con población 2. Si cada una toma 3-5 min en tu L40S, el costo agregado de verificación es de 1.5 a 2.5 horas adicionales sobre las ~10 horas estimadas de refactor.
