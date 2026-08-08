# Mapa de ruta de refactorización FINAL por etapas atómicas — MoQ-NAS

> Derivado de `REFACTOR_GUIDE.md` (2026-05-04) y `REFACTOR_ROADMAP.md`, **corregido contra el estado real del código** (validación empírica 2026-06-09, rama `update-2026`).
> Reordena y subdivide en **33 etapas atómicas**, cada una con red de seguridad propia (pre-verificación, post-verificación, métrica de éxito).

---

## Variables de entorno requeridas

Todos los scripts de este roadmap usan variables de entorno en lugar de rutas hardcodeadas. Exportarlas una vez por sesión antes de ejecutar cualquier etapa:

```bash
export PROJECT_ROOT="/home/diegopaez/MoQ-NAS"   # raíz del repo  (sustituye el antiguo <ruta_proyecto>)
export DATA_PATH="/ruta/a/tus/datasets"         # datos de entrada (sustituye el antiguo <ruta_datos>)
export REFACTOR_SEED=42                          # semilla global para baselines y Bloque D
```

Convenciones:
- `--data_path "$DATA_PATH"` en todo smoke run.
- En tests que insertan el path del proyecto: `python -c "import os,sys; sys.path.insert(0, os.environ['PROJECT_ROOT']); ..."`.
- `--seed "$REFACTOR_SEED"` se añade a partir de la Etapa 0.5 (antes de esa etapa el flag no existe).

---

## Hallazgos de la validación empírica (resumen)

Correcciones incorporadas respecto al roadmap original:

1. **Cero drift de números de línea.** Todas las referencias de `REFACTOR_GUIDE.md` coinciden con el código actual.
2. **Drift de naming (crítico).** Los operadores en `nsga2.py`/`moqnas.py` son **públicos** (`dominates`, `fast_nondominated_sort`, `crowding_distance`, `compute_hypervolume_mixed`, **sin** guion bajo). Los de `nsga3.py`/`moead.py` **sí** son privados (`_simplex_lattice`, `_build_reference_directions`, `_to_minimization`). El atributo de sentidos es `self.objective_senses` (lista `['max','min','min']`), no `self.objectives_info`. Todos los scripts del Bloque D fueron corregidos en consecuencia.
3. **Duplicaciones idénticas vs divergentes:**
   - `dominates`, `fast_nondominated_sort`, `crowding_distance` (nsga2≡moqnas): **idénticas** → canónica = nsga2.
   - `compute_hypervolume_mixed` (nsga2≡moqnas): **idénticas**; **`utils/helpers.py:852` DIVERGE** (hardcodea `f[:,0]=-f[:,0]`, no voltea `ref_point`) → canónica = clase.
   - `_simplex_lattice`, `_to_minimization` (nsga3≡moead): **idénticas** → canónica = nsga3.
   - `_build_reference_directions`: **DIVERGEN** (nsga3 no poda; moead poda con `np.random.choice`) → **NO se consolida**; permanece específico por algoritmo.
4. **Ciclo `core ↔ algorithms`:**
   - `core/config.py:15` → `QChromosome*` es **import muerto** (Caso 1) → borrado trivial.
   - `base_ga.py:7-8` → `cfg.*`/`evaluation.*` es **uso real** (Caso 3, líneas 670/683, función de orquestación) → requiere mover wiring a `run_all_evolution.py`. Riesgo alto.
5. **Sin determinismo de semillas.** No hay `manual_seed`/`np.random.seed` en ningún módulo. El Bloque D end-to-end no puede comparar HV bit-a-bit. → **Etapa 0.5 (fijar semillas) obligatoria**; verificación de operadores por **paridad sintética determinista + tolerancia** en end-to-end.

---

## Erratum — desviaciones encontradas al ejecutar las etapas 0/0.5 (2026-06-10)

Correcciones que aplican a TODOS los scripts de verificación de etapas posteriores:

1. **Los `config0.txt` del roadmap ya no parsean.** `core/config.py` exige claves nuevas (`mo_crossover_strategy`, `quantum_update_config`, etc.) ausentes en los configs viejos. Usar `config_files/config_files_cifar_mo/config0_3.txt` en TODOS los smoke runs (la ruta cambiará con los renombrados A.6/A.7).
2. **moqnas/qnas ignoraban `--num_generations`/`--population_size`** (leían solo el config). El baseline moqnas de 0.5 usó `.refactor_baseline/config_moqnas_baseline.txt` (config0_3 con `max_generations: 1`). Desde la etapa 0.6, `--num_generations` SÍ sobreescribe `max_generations` para qnas/moqnas; `--population_size` sigue sin aplicar (la población es `num_quantum_ind × repetition`).
3. **No hay líneas "hypervolume" en logs de 1 generación** — los grep de HV del roadmap no encuentran nada. Comparar las líneas por candidato `best_accuracy=` / `total_params=` en su lugar.
4. **La accuracy entrenada NO era reproducible en 0.5 (~1-3 pp de spread) aunque las arquitecturas/`total_params` sí eran bit-exactas.** Causa raíz (diagnosticada con corridas `threads: 1`): `GenericDataLoader.__init__` resembraba los RNG globales (`random`, `torch`) con `int(time())`, destruyendo `--seed`; numpy no se tocaba, por eso las arquitecturas eran estables. Contribuía además el interleaving de workers y el avance del generator del DataLoader entre candidatos del mismo proceso. **Resuelto en la etapa 0.6** — accuracy por candidato es bit-exacta entre corridas e independiente del número de threads. Los baselines de 0.5 (`baseline_*.log`) son anteriores al fix: para comparaciones end-to-end del Bloque D usar los logs de `.refactor_baseline/expB_run1` (post-0.6) o regenerar baselines, comparando accuracy por candidato (bit-exacta), no solo arquitecturas.

---

## Principios operativos

1. **Una etapa, un commit.** Si la post-verificación falla, `git restore .` y reanálisis; nunca se acumulan cambios.
2. **Red de seguridad antes de tocar nada.** Etapa 0 + Etapa 0.5 establecen la línea base reproducible.
3. **Verificación numérica explícita en operadores MOEA.** Antes de borrar cualquier duplicado se ejecuta un script de paridad sintético (semilla local `np.random.seed`) que compara módulo nuevo vs duplicado existente con los mismos inputs. Estos checks son deterministas e independientes de GPU.
4. **Reordenamiento sobre la guía.** `settings.py` (Bloque A) antes del renombrado, para no tocarlo dos veces.
5. **Política de documentación (opción A).** Docstrings y refactor estructural van en el **mismo commit** por etapa. Código nuevo: inglés, estilo NumPy (Parameters/Returns/Raises; Examples solo si el uso no es evidente). Código que se mueve: conserva su docstring original; si no lo tiene, se añade NumPy en el mismo commit.

---

## ETAPA 0 — línea base git + mapa de imports

**Objetivo atómico.** Estado de referencia git y mapa de imports. (Los baselines de evolución se generan en 0.5, tras fijar semillas; sin semilla no sirven como referencia del Bloque D.)

**Script de pre-verificación.**
```bash
git status                                          # debe estar limpio
git log -1 --oneline                                # registrar SHA base
git checkout -b refactor/update-2026-staged
```

**Instrucciones de refactorización.** No se modifica código. Generar el mapa de imports en `.refactor_baseline/` (ignorada por git):
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

**Script de post-verificación.**
```bash
test -f .refactor_baseline/imports.json && echo "OK baseline imports"
grep -qxF '.refactor_baseline/' .gitignore || echo ".refactor_baseline/" >> .gitignore
```

**Métrica de éxito.** `imports.json` existe. `git add .gitignore && git commit -m "refactor(0): import map and staging branch"`.

---

## ETAPA 0.5 — fijar semillas globales y generar baselines reproducibles  *(NUEVA, obligatoria)*

**Objetivo atómico.** Introducir control de semillas determinista y generar los baselines seeded que el Bloque D usará como referencia. Sin esto, los `diff` de hipervolumen del Bloque D fallarían por ruido estocástico, no por el refactor.

**Script de pre-verificación.**
```bash
grep -rn "manual_seed\|np.random.seed\|random.seed" run_all_evolution.py algorithms/ core/ || echo "OK: confirmado, no hay semillas hoy"
```

**Instrucciones de refactorización.**
1. Crear `utils/seeding.py` con `set_global_seeds(seed)`.
2. En `run_all_evolution.py`: añadir arg `--seed` (default desde `REFACTOR_SEED` o 42) e invocar `set_global_seeds(args['seed'])` al inicio de `main()`, antes de cualquier `np.random`/torch.
3. Propagar la llamada al inicio de la función de orquestación de `algorithms/ga/base_ga.py` si el entry point pasa por ahí.

```python
# utils/seeding.py
import os, random
import numpy as np
import torch


def set_global_seeds(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all RNGs used by the evolutionary search for reproducible runs.

    Parameters
    ----------
    seed : int, default 42
        Seed applied to ``random``, ``numpy`` and ``torch`` (CPU and CUDA).
    deterministic : bool, default True
        If True, force cuDNN into deterministic mode and disable benchmark
        autotuning. Trades some GPU throughput for run-to-run stability.

    Notes
    -----
    Even with ``deterministic=True``, a few CUDA kernels lack deterministic
    implementations, so end-to-end hypervolume may still differ at the
    ~1e-6 level across runs. Operator correctness in Block D is therefore
    verified with synthetic parity scripts, not bit-exact end-to-end diffs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

4. Generar baselines seeded:
```bash
for spec in "nsga2:config_files/config_files_cifar/config0.txt" \
            "moqnas:config_files/config_files_cifar_mo/config0.txt" \
            "nsga3:config_files/config_files_cifar/config0.txt" \
            "moead:config_files/config_files_cifar/config0.txt"; do
  algo="${spec%%:*}"; cfg="${spec##*:}"
  python run_all_evolution.py --algo "$algo" --num_generations 1 --population_size 2 \
    --seed "$REFACTOR_SEED" --config_file "$cfg" \
    --experiment_path ".refactor_baseline/baseline_$algo" \
    --data_path "$DATA_PATH" --dataset cifar10 \
    --config_path_dataset configs/cifar10.yaml --log_level INFO \
    2>&1 | tee ".refactor_baseline/baseline_$algo.log"
done
```

**Documentación añadida.** `utils/seeding.py`: docstring de módulo + docstring NumPy de `set_global_seeds` (Parameters, Notes con caveat cuDNN).

**Script de post-verificación.**
```bash
python -c "from utils.seeding import set_global_seeds; set_global_seeds(42); print('OK seed import')"
for a in nsga2 moqnas nsga3 moead; do test -f ".refactor_baseline/baseline_$a.log" && echo "OK baseline $a"; done
# Reproducibilidad: dos corridas con misma semilla deben dar el mismo HV (salvo caveat cuDNN)
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --seed "$REFACTOR_SEED" --config_file config_files/config_files_cifar/config0.txt \
  --experiment_path /tmp/seedcheck --data_path "$DATA_PATH" --dataset cifar10 \
  --config_path_dataset configs/cifar10.yaml --log_level INFO 2>&1 | tee /tmp/seedcheck.log
diff <(grep -i hypervolume .refactor_baseline/baseline_nsga2.log | tail -1) \
     <(grep -i hypervolume /tmp/seedcheck.log | tail -1) && echo "OK reproducible" \
  || echo "AVISO: difiere -> documentar caveat cuDNN y usar tolerancia en Bloque D"
```

**Métrica de éxito.** `set_global_seeds` importable; 4 baselines seeded generados; dos corridas con misma semilla coinciden (o se documenta el caveat cuDNN y la tolerancia a usar). `git commit -m "refactor(0.5): add global seeding and reproducible baselines"`.

---

## ETAPA 0.6 — siembra determinista por candidato y override de generaciones  *(NUEVA, surgida del erratum 0.5)*

**Objetivo atómico.** Hacer la accuracy entrenada bit-exacta entre corridas con la misma semilla, independiente del scheduling de threads, y hacer que `--num_generations` aplique a qnas/moqnas. Sin esto, los checks end-to-end del Bloque D solo podrían comparar arquitecturas, no fitness.

**Diagnóstico previo (experimento A).** Dos corridas nsga2 secuenciales con `threads: 1` y misma semilla seguían difiriendo en accuracy (arquitecturas bit-exactas). Causa raíz: `core/cnn/input.py` resembraba `random`/`torch` globales con `int(time())` en `GenericDataLoader.__init__`.

**Instrucciones de refactorización.**
1. `core/cnn/input.py`: eliminar el reseed global con `time()`; el loader solo usa generadores locales (`split_seed`/`loader_seed`, fallback `params['seed']`).
2. `utils/seeding.py`: añadir `seed_candidate(global_seed, generation, candidate_id)` que resiembra `random`/`numpy`/`torch` con `(global_seed + 100_003·generation + candidate_id) % 2³¹` y devuelve la semilla derivada.
3. `core/evaluation.py`: en `run_individuals`, antes de cada `master.fitness`, llamar `seed_candidate(...)` y resembrar `train_loader.generator` con la semilla derivada (el loader es compartido por los candidatos del mismo proceso; su generator avanza entre candidatos).
4. `core/config.py`: añadir `'seed'` a `train_override_keys` para que el seed CLI llegue a `train_spec` (y de ahí a `EvalPopulation`).
5. `run_all_evolution.py`: `--num_generations` con default `None`; si se pasa y el algo es qnas/moqnas, sobreescribe `config.QNAS_spec['max_generations']`; si no se pasa, GA-family usa 50 y qnas/moqnas usan el config.

**Script de post-verificación (experimento B).**
```bash
# 3 corridas nsga2 seed=42 pop=4: run1/run2 con threads=4, run3 con threads=2
bash .refactor_baseline/expB.sh
# debe imprimir: run1 vs run2 BIT-EXACTO, run1 vs run3 BIT-EXACTO
# override moqnas: con config0_3 (max_generations: 150) + --num_generations 1
# debe loguear "Overriding config max_generations (150) with --num_generations 1"
# y terminar tras 1 generación.
```

**Métrica de éxito.** Accuracy por candidato bit-exacta entre corridas con la misma semilla e independiente de `threads`; moqnas respeta `--num_generations`. `git commit -m "refactor(0.6): per-candidate deterministic seeding and qnas/moqnas generation override"`.

---

## Bloque A — settings y rutas

### ETAPA A.1 — crear `settings.py` apuntando a rutas antiguas

**Objetivo atómico.** Centralizar `CFG_OBJ_PATH` y `TRAIN_TIMEOUT`, sin tocar consumidores aún.

**Script de pre-verificación.**
```bash
test ! -f settings.py && echo "OK, settings.py no existe aún"
python -c "import os; assert os.path.isfile('configs/cfg_obj.json')"
grep -n "TRAIN_TIMEOUT" core/cnn/trainer.py
```

**Instrucciones de refactorización.** Crear `settings.py` en la raíz con `PROJECT_ROOT`, `DATASET_CONFIGS_DIR` (apuntando a `configs/`, ruta antigua intencional, comentar `# se actualiza en A.5`), `CFG_OBJ_PATH`, `TRAIN_TIMEOUT = 5400`.

**Documentación añadida.** Docstring de módulo NumPy en `settings.py`; comentario inline por constante (unidad/propósito de `TRAIN_TIMEOUT`).

**Script de post-verificación.**
```bash
python -c "from settings import CFG_OBJ_PATH, TRAIN_TIMEOUT; import os; assert os.path.isfile(CFG_OBJ_PATH), CFG_OBJ_PATH; assert TRAIN_TIMEOUT == 5400; print('OK')"
test ! -d logs && echo "OK, no side-effects"
```

**Métrica de éxito.** `OK` en ambos; ningún `.py` existente modificado. `git commit -m "refactor(A.1): add settings.py"`.

---

### ETAPA A.2 — migrar lectura de `cfg_obj.json` en `nsga2.py`

**Objetivo atómico.** Reemplazar `"configs/cfg_obj.json"` (`nsga2.py:48`) por `CFG_OBJ_PATH`. Cambio quirúrgico de 2 líneas.

**Script de pre-verificación.**
```bash
grep -n "configs/cfg_obj.json" algorithms/ga/nsga2.py
python -c "from algorithms.ga.nsga2 import NSGA2; print('import OK')"
```

**Instrucciones de refactorización.** Añadir `from settings import CFG_OBJ_PATH` y reemplazar `open("configs/cfg_obj.json", "r")` por `open(CFG_OBJ_PATH, "r")`. Nada más.

**Script de post-verificación.**
```bash
grep -n "configs/cfg_obj.json" algorithms/ga/nsga2.py && echo "FAIL" || echo "OK"
python -c "from algorithms.ga.nsga2 import NSGA2; print('OK')"
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --seed "$REFACTOR_SEED" --config_file config_files/config_files_cifar/config0.txt \
  --experiment_path /tmp/test_A2 --data_path "$DATA_PATH" --dataset cifar10 \
  --config_path_dataset configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** Smoke run sin `FileNotFoundError`, frente Pareto con cardinalidad igual al baseline nsga2. `git commit -m "refactor(A.2): nsga2 reads CFG_OBJ_PATH from settings"`.

---

### ETAPA A.3 — migrar lectura de `cfg_obj.json` en `moqnas.py`

**Objetivo atómico.** Mismo cambio de A.2 en `moqnas.py:221` (nota: el log informativo está en `moqnas.py:219`; actualizar también el texto del log si menciona la ruta).

**Script de pre-verificación.**
```bash
grep -n "configs/cfg_obj.json" algorithms/qnas/moqnas.py
```

**Instrucciones de refactorización.** Idéntico a A.2 en `moqnas.py`; opcionalmente actualizar el mensaje de log de la línea 219 para que refleje `CFG_OBJ_PATH`.

**Script de post-verificación.**
```bash
grep -n "configs/cfg_obj.json" algorithms/qnas/moqnas.py && echo "FAIL" || echo "OK"
python -c "from algorithms.qnas.moqnas import MoQNAS; print('OK')"
python run_all_evolution.py --algo moqnas --num_generations 1 --population_size 2 \
  --seed "$REFACTOR_SEED" --config_file config_files/config_files_cifar_mo/config0.txt \
  --experiment_path /tmp/test_A3 --data_path "$DATA_PATH" --dataset cifar10 \
  --config_path_dataset configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** Smoke run exitoso, cardinalidad de frente igual al baseline. `git commit -m "refactor(A.3): moqnas reads CFG_OBJ_PATH from settings"`.

---

### ETAPA A.4 — migrar `TRAIN_TIMEOUT` en `trainer.py`

**Objetivo atómico.** Eliminar el literal `TRAIN_TIMEOUT = 5400` (`trainer.py:27`) y leerlo de `settings`. El uso interno (`trainer.py:297`) no se toca.

**Script de pre-verificación.**
```bash
grep -n "TRAIN_TIMEOUT" core/cnn/trainer.py
python -c "from core.cnn.trainer import BaseTrainer; print('import OK')"
```

**Instrucciones de refactorización.** Reemplazar la línea 27 por `from settings import TRAIN_TIMEOUT`.

**Script de post-verificación.**
```bash
python -c "from core.cnn import trainer; assert trainer.TRAIN_TIMEOUT == 5400; print('OK')"
```

**Métrica de éxito.** `trainer.TRAIN_TIMEOUT == 5400`. `git commit -m "refactor(A.4): trainer reads TRAIN_TIMEOUT from settings"`.

---

### ETAPA A.5 — renombrar `configs/` a `dataset_configs/`

**Objetivo atómico.** Renombrar la carpeta y actualizar las 3 ubicaciones que la referencian (settings.py, las refs `config_path_dataset: configs/...` dentro de los `.txt`, y los `.sh` si aplica).

**Script de pre-verificación.**
```bash
grep -rln "configs/cifar\|configs/medmnist\|configs/cfg_obj.json\|configs/atleta\|configs/face\|configs/person\|configs/oct\|configs/organ\|configs/path\|configs/tissue" \
  config_files/ algorithms/ core/ utils/ dataset_utils/ scripts/ *.py *.sh > /tmp/refs_pre.txt
cat /tmp/refs_pre.txt
```

**Instrucciones de refactorización.**
1. `git mv configs dataset_configs`
2. `settings.py`: `DATASET_CONFIGS_DIR = os.path.join(PROJECT_ROOT, "dataset_configs")` (quitar comentario temporal).
3. Reemplazar `configs/` → `dataset_configs/` solo en las refs de `/tmp/refs_pre.txt`. **Cuidado:** no tocar `config_files/` (empieza igual).

**Script de post-verificación.**
```bash
test -d dataset_configs && test ! -d configs && echo "OK rename"
grep -rln "configs/cifar\|configs/cfg_obj.json" . --include="*.py" --include="*.txt" --include="*.sh" \
  | grep -v dataset_configs && echo "FAIL, quedan refs" || echo "OK"
python -c "from settings import CFG_OBJ_PATH; import os; assert os.path.isfile(CFG_OBJ_PATH); print(CFG_OBJ_PATH)"
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --seed "$REFACTOR_SEED" --config_file config_files/config_files_cifar/config0.txt \
  --experiment_path /tmp/test_A5 --data_path "$DATA_PATH" --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** Smoke run con la ruta nueva. `git commit -m "refactor(A.5): rename configs/ to dataset_configs/"`.

---

### ETAPA A.6 — renombrar `config_files/` a `experiment_configs/` y subcarpetas

**Objetivo atómico.** Renombrar la carpeta raíz y sus 6 subcarpetas (`config_files_cifar`→`cifar`, `config_files_cifar_mo`→`cifar_mo`, `config_files_fairness`→`fairness`, `config_files_med`→`medmnist`, `config_files_med_base`→`medmnist_base`, `config_files_med_mo`→`medmnist_mo`). No cambiar extensiones. **Preservar** el notebook `config_probs.ipynb`.

**Script de pre-verificación.**
```bash
ls config_files/
find config_files -name "*.ipynb"   # registrar config_probs.ipynb
grep -rln "config_files" *.sh scripts/ algorithms/ core/ utils/ dataset_utils/ *.py > /tmp/refs_cf_pre.txt
cat /tmp/refs_cf_pre.txt
```

**Instrucciones de refactorización.** `git mv` carpeta raíz y las 6 subcarpetas. No tocar archivos individuales aún.

**Script de post-verificación.**
```bash
test -d experiment_configs && test ! -d config_files && echo "OK"
ls experiment_configs/
find experiment_configs -name "config_probs.ipynb" && echo "OK notebook preservado"
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --seed "$REFACTOR_SEED" --config_file experiment_configs/cifar/config0.txt \
  --experiment_path /tmp/test_A6 --data_path "$DATA_PATH" --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** Smoke run con `experiment_configs/cifar/config0.txt`. `git commit -m "refactor(A.6): rename config_files/ to experiment_configs/"`.

---

### ETAPA A.7 — renombrar extensiones `.txt` a `.yaml`

**Objetivo atómico.** Renombrar los 31 `.txt` dentro de `experiment_configs/**` a `.yaml`. `core/config.py` usa `yaml.safe_load()` (agnóstico a extensión). El notebook `.ipynb` no se toca.

**Script de pre-verificación.**
```bash
find experiment_configs -name "*.txt" | wc -l   # debe ser 31
find experiment_configs -name "*.txt" > /tmp/txt_files.txt
```

**Instrucciones de refactorización.**
```bash
find experiment_configs -name "*.txt" -exec bash -c 'git mv "$1" "${1%.txt}.yaml"' _ {} \;
```

**Script de post-verificación.**
```bash
test "$(find experiment_configs -name '*.txt' | wc -l)" -eq 0 && echo "OK, 0 .txt"
test "$(find experiment_configs -name '*.yaml' | wc -l)" -eq 31 && echo "OK, 31 .yaml"
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --seed "$REFACTOR_SEED" --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_A7 --data_path "$DATA_PATH" --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** Cero `.txt`, 31 `.yaml`, smoke run con `.yaml`. `git commit -m "refactor(A.7): rename experiment configs .txt to .yaml"`.

---

### ETAPA A.8 — actualizar los 6 scripts `.sh`

**Objetivo atómico.** Actualizar rutas en `run_ea_1.sh`, `run_fair_mo.sh`, `run_fairness_baseline.sh`, `run_moqnas_1.sh`, `run_qnas_1.sh`, `run_retrain.sh` a `experiment_configs/.../config*.yaml` y `dataset_configs/...`.

**Script de pre-verificación.**
```bash
grep -n "config_files\|configs/\|\.txt" *.sh
```

**Instrucciones de refactorización.** Buscar/reemplazar en cada `.sh`: `config_files/config_files_*` → `experiment_configs/*`, `.txt` → `.yaml`, `configs/` → `dataset_configs/`.

**Script de post-verificación.**
```bash
grep -n "config_files\|\.txt" *.sh && echo "FAIL" || echo "OK"
for f in run_*.sh; do bash -n "$f" && echo "OK $f"; done
```

**Métrica de éxito.** Sin refs residuales, todos pasan `bash -n`. `git commit -m "refactor(A.8): update shell scripts to new config paths"`.

---

## Bloque B — eliminar side-effects en import

### ETAPA B.1 — sanear `core/cnn/master.py`

**Objetivo atómico.** Eliminar el bloque (líneas 23-29) que crea `logs/` al hacer `import master`.

**Script de pre-verificación.**
```bash
rm -rf /tmp/B1_test && mkdir /tmp/B1_test && cd /tmp/B1_test && \
  python -c "import os,sys; sys.path.insert(0, os.environ['PROJECT_ROOT']); from core.cnn import master" && \
  ls -la /tmp/B1_test   # si aparece logs/, el side-effect está confirmado
```

**Instrucciones de refactorización.** Reemplazar las líneas 23-29 por:
```python
import logging
LOGGER = logging.getLogger(__name__)
```
No tocar el resto. El logger concreto se configura aguas abajo cuando se conoce `experiment_path`.

**Documentación añadida.** Ninguna función nueva; solo comentario inline explicando que la configuración del handler se difiere a la instancia.

**Script de post-verificación.**
```bash
rm -rf /tmp/B1_test && mkdir /tmp/B1_test && cd /tmp/B1_test && \
  python -c "import os,sys; sys.path.insert(0, os.environ['PROJECT_ROOT']); from core.cnn import master" && \
  test ! -d logs && echo "OK, no side-effect"
```

**Métrica de éxito.** Importar `master` desde CWD vacío no crea nada. `git commit -m "refactor(B.1): remove import-time side-effects in master.py"`.

---

### ETAPA B.2 — sanear `core/cnn/trainer.py`

**Objetivo atómico.** Idéntico a B.1 en `trainer.py` líneas 29-32. (Nota: `TRAIN_TIMEOUT` ya migró a settings en A.4; aquí solo el bloque de logger.)

**Script de pre-verificación.**
```bash
rm -rf /tmp/B2_test && mkdir /tmp/B2_test && cd /tmp/B2_test && \
  python -c "import os,sys; sys.path.insert(0, os.environ['PROJECT_ROOT']); from core.cnn import trainer"; \
  ls -la /tmp/B2_test
```

**Instrucciones de refactorización.** Reemplazar el bloque de logger por `import logging; LOGGER = logging.getLogger(__name__)`.

**Script de post-verificación.**
```bash
rm -rf /tmp/B2_test && mkdir /tmp/B2_test && cd /tmp/B2_test && \
  python -c "import os,sys; sys.path.insert(0, os.environ['PROJECT_ROOT']); from core.cnn import trainer" && \
  test ! -d logs && echo "OK"
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --seed "$REFACTOR_SEED" --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_B2 --data_path "$DATA_PATH" --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** Import sin side-effects y smoke run sin error de logger. `git commit -m "refactor(B.2): remove import-time side-effects in trainer.py"`.

---

## Bloque C — dividir `utils/helpers.py`

Estrategia: 5 sub-etapas crean los módulos nuevos sin tocar `helpers.py`; la sexta lo convierte en facade. Preserva todos los `from utils.helpers import X` hasta el final.

**Política de documentación del bloque.** Cada función movida **conserva su docstring original**. Las que hoy no tienen docstring reciben uno NumPy nuevo en el mismo commit (ver subsecciones "Documentación añadida"). Cada módulo nuevo recibe docstring de módulo en inglés.

### ETAPA C.1 — crear `utils/io.py`

**Objetivo atómico.** Mover las funciones de I/O (sección 3a de la guía) conservando firmas y cuerpos. **Exclusión:** no mover `load_evolved_data` standalone (se elimina en F.2).

**Script de pre-verificación.**
```bash
python -c "from utils.helpers import load_yaml, save_pkl, load_pkl, update_yaml_file, backup_cache, load_cache, load_log_params_evolution, load_pareto_history, load_history_from_json, save_history_to_json, load_retrain_results, create_info_file, save_results_file; print('OK pre')"
```

**Instrucciones de refactorización.** Crear `utils/io.py` con esas funciones y sus imports (yaml, pickle, json, pathlib, os). `helpers.py` queda intacto (duplicación temporal).

**Documentación añadida.** Docstring de módulo en `utils/io.py`. Docstrings NumPy nuevos para las funciones de I/O que hoy carezcan de ellos (revisar `_deep_merge`, `_atomic_write_yaml`, `create_info_file`, `save_results_file`).

**Script de post-verificación.**
```bash
python -c "from utils.io import load_yaml, save_pkl, load_pkl, update_yaml_file, backup_cache, load_cache, load_log_params_evolution, load_pareto_history, load_history_from_json, save_history_to_json, load_retrain_results, create_info_file, save_results_file; print('OK post')"
python -c "
from utils.helpers import load_yaml as old
from utils.io import load_yaml as new
r1 = old('dataset_configs/cifar10.yaml'); r2 = new('dataset_configs/cifar10.yaml')
assert r1 == r2, 'divergencia en load_yaml'; print('OK paridad')
"
```

**Métrica de éxito.** Ambos imports + paridad de `load_yaml`. `git commit -m "refactor(C.1): add utils/io.py (helpers.py untouched)"`.

---

### ETAPA C.2 — crear `utils/logging_utils.py`

**Objetivo atómico.** Mover `init_log` (`helpers.py:396`).

**Script de pre-verificación.**
```bash
python -c "from utils.helpers import init_log; print(init_log)"
```

**Instrucciones de refactorización.** Crear `utils/logging_utils.py` con `init_log` y sus imports (`logging`, `os`).

**Documentación añadida.** Docstring de módulo; `init_log` conserva su docstring (añadir NumPy si no lo tiene).

**Script de post-verificación.**
```bash
python -c "from utils.logging_utils import init_log; print('OK')"
rm -rf /tmp/C2_test && mkdir /tmp/C2_test && cd /tmp/C2_test && \
  python -c "import os,sys; sys.path.insert(0, os.environ['PROJECT_ROOT']); from utils.logging_utils import init_log" && \
  test ! -d logs && echo "OK sin side-effects"
```

**Métrica de éxito.** Import sin crear `logs/`. `git commit -m "refactor(C.2): add utils/logging_utils.py"`.

---

### ETAPA C.3 — crear `utils/experiment.py`

**Objetivo atómico.** Mover `natural_key`, `check_file_exists`, `check_files`, `delete_old_dirs`, `delete_old_dirs_v2`, `calculate_time`.

**Script de pre-verificación.**
```bash
python -c "from utils.helpers import natural_key, check_file_exists, check_files, delete_old_dirs, delete_old_dirs_v2, calculate_time; print('OK')"
```

**Instrucciones de refactorización.** Crear `utils/experiment.py` con las funciones y dependencias (re, os, shutil, datetime, time).

**Documentación añadida.** Docstring de módulo; docstring NumPy nuevo para `natural_key` (uso no evidente → incluir Examples breve).

**Script de post-verificación.**
```bash
python -c "from utils.experiment import natural_key, check_file_exists, check_files, delete_old_dirs, delete_old_dirs_v2, calculate_time; print('OK')"
python -c "
from utils.helpers import natural_key as old
from utils.experiment import natural_key as new
assert old('exp_10') == new('exp_10') and old('exp_2') == new('exp_2'); print('OK paridad')
"
```

**Métrica de éxito.** Imports + paridad `natural_key`. `git commit -m "refactor(C.3): add utils/experiment.py"`.

---

### ETAPA C.4 — crear `utils/dataset.py`

**Objetivo atómico.** Mover `download_dataset`, `setup_dataset_info`, `dataset_cache`, `_validate_dataset_info`.

**Script de pre-verificación.**
```bash
python -c "from utils.helpers import download_dataset, setup_dataset_info; print('OK')"
```

**Instrucciones de refactorización.** Crear `utils/dataset.py` con las funciones y sus imports (torchvision, medmnist, etc.). Mantener `dataset_cache` como global del módulo.

**Documentación añadida.** Docstring de módulo; docstring NumPy nuevo para `_validate_dataset_info` y para el global `dataset_cache` (comentario explicando su rol).

**Script de post-verificación.**
```bash
python -c "from utils.dataset import download_dataset, setup_dataset_info, dataset_cache; print('OK')"
```

**Métrica de éxito.** Imports sin side-effects de descarga. `git commit -m "refactor(C.4): add utils/dataset.py"`.

---

### ETAPA C.5 — crear `utils/visualization.py`

**Objetivo atómico.** Mover las funciones de plotting/agregación (sección 3e de la guía).

**⚠️ Corrección.** La standalone `compute_hypervolume_mixed` (`helpers.py:852`) es la **versión DIVERGENTE** (hardcodea `f[:,0]=-f[:,0]`, no voltea `ref_point`). Se mueve aquí **marcada explícitamente como no-canónica** y se eliminará/sustituirá por la canónica parametrizada en D.6-cleanup. No se usa como referencia de paridad.

**Script de pre-verificación.**
```bash
python -c "from utils.helpers import plot_training_history, agg_results, plot_hypervolume_comparison, plot_pareto_evolution; print('OK')"
```

**Instrucciones de refactorización.** Crear `utils/visualization.py` con las funciones y sus imports (matplotlib, plotly, numpy, pandas, GPUtil). Anotar la standalone con `# NON-CANONICAL: hardcodes col0; replaced by algorithms.pareto.hypervolume in D-cleanup`.

**Documentación añadida.** Docstring de módulo; docstrings NumPy nuevos para los plots sin docstring; nota de deprecación en el docstring de la standalone.

**Script de post-verificación.**
```bash
python -c "from utils.visualization import plot_training_history, agg_results, plot_hypervolume_comparison, plot_pareto_evolution, compute_hypervolume_mixed; print('OK')"
```

**Métrica de éxito.** Imports funcionales. `git commit -m "refactor(C.5): add utils/visualization.py"`.

---

### ETAPA C.6 — convertir `utils/helpers.py` en facade

**Objetivo atómico.** Reemplazar `helpers.py` por re-exports desde los 5 módulos. Cero consumidores rotos. Excluir `load_evolved_data`.

**Script de pre-verificación.**
```bash
grep -rn "from utils.helpers import\|from utils import helpers" . --include="*.py" > /tmp/helpers_consumers.txt
wc -l /tmp/helpers_consumers.txt
```

**Instrucciones de refactorización.** Reemplazar el contenido por re-exports (sección 3f de la guía), sin `load_evolved_data`.

**Documentación añadida.** Comentario de cabecera explicando que es una facade de compatibilidad temporal (deprecación formal en F.3).

**Script de post-verificación.**
```bash
python -c "
from utils.helpers import (load_yaml, save_pkl, load_pkl, init_log, download_dataset,
  setup_dataset_info, delete_old_dirs_v2, check_files, calculate_time,
  plot_training_history, plot_hypervolume_comparison, compute_hypervolume_mixed,
  agg_results, backup_cache, load_cache); print('OK facade')"
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --seed "$REFACTOR_SEED" --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_C6 --data_path "$DATA_PATH" --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO 2>&1 | tee /tmp/test_C6.log
# HV dentro de tolerancia respecto al baseline seeded
diff <(grep -i hypervolume .refactor_baseline/baseline_nsga2.log | tail -1) \
     <(grep -i hypervolume /tmp/test_C6.log | tail -1) || echo "AVISO: revisar tolerancia"
```

**Métrica de éxito.** Smoke run OK, HV dentro de tolerancia del baseline; `wc -l utils/helpers.py` < 30. `git commit -m "refactor(C.6): convert utils/helpers.py to facade"`.

---

## Bloque D — operadores Pareto

Estrategia: crear `algorithms/pareto/*` sin borrar duplicados (D.1–D.5). Tras cada creación, **paridad sintética determinista** (semilla local) contra el duplicado. Solo entonces migrar cada algoritmo (D.6–D.9). La verificación de corrección descansa en las paridades sintéticas; los checks end-to-end usan **tolerancia**, no bit-exacto (caveat cuDNN documentado en 0.5).

**Naming corregido:** nsga2/moqnas exponen métodos **públicos** (`dominates`, `fast_nondominated_sort`, `crowding_distance`, `compute_hypervolume_mixed`); nsga3/moead exponen **privados** (`_simplex_lattice`, `_to_minimization`, `_build_reference_directions`). El atributo de sentidos es `self.objective_senses` (lista).

### ETAPA D.1 — crear `algorithms/pareto/dominance.py`

**Objetivo atómico.** Crear `dominance.py` con `dominates` y `fast_nondominated_sort`, copiando de `nsga2.py:249`/`:265` (canónicas; idénticas a moqnas), parametrizadas por `objective_senses: list`.

**Script de pre-verificación.**
```bash
test ! -d algorithms/pareto && echo "OK, no existe aún"
grep -n "def dominates\|def fast_nondominated_sort" algorithms/ga/nsga2.py algorithms/qnas/moqnas.py
```

**Instrucciones de refactorización.** Crear `algorithms/pareto/__init__.py` (vacío) y `algorithms/pareto/dominance.py`. No tocar nsga2/moqnas.

**Documentación añadida.** Docstring de módulo; docstrings NumPy nuevos para `dominates(a, b, objective_senses)` y `fast_nondominated_sort(fits, objective_senses)` (Parameters incluyendo `objective_senses`, Returns).

**Script de post-verificación.**
```bash
python -c "from algorithms.pareto.dominance import dominates, fast_nondominated_sort; print('OK import')"
python -c "
import numpy as np
from algorithms.pareto.dominance import dominates as new_dom
from algorithms.ga import nsga2
np.random.seed(0)
fits = np.random.rand(10, 3); senses = ['max','min','min']
inst = nsga2.NSGA2.__new__(nsga2.NSGA2); inst.objective_senses = senses
for i in range(10):
  for j in range(10):
    assert new_dom(fits[i], fits[j], senses) == inst.dominates(fits[i], fits[j])
print('OK dominates paridad')
"
```

**Métrica de éxito.** Paridad booleana sobre 100 comparaciones. `git commit -m "refactor(D.1): add algorithms/pareto/dominance.py"`.

---

### ETAPA D.2 — crear `algorithms/pareto/diversity.py`

**Objetivo atómico.** Crear `crowding_distance` (de `nsga2.py:300`; idéntica a moqnas).

**Script de pre-verificación.**
```bash
grep -n "def crowding_distance" algorithms/ga/nsga2.py algorithms/qnas/moqnas.py
```

**Instrucciones de refactorización.** Copiar `crowding_distance(fits, front)` a `algorithms/pareto/diversity.py`.

**Documentación añadida.** Docstring de módulo; preservar el docstring existente de `crowding_distance`.

**Script de post-verificación.**
```bash
python -c "
import numpy as np
from algorithms.pareto.diversity import crowding_distance as new_cd
from algorithms.ga import nsga2
np.random.seed(1)
fits = np.random.rand(20, 3); front = list(range(20))
inst = nsga2.NSGA2.__new__(nsga2.NSGA2)
assert np.allclose(new_cd(fits, front), inst.crowding_distance(fits, front), equal_nan=True)
print('OK crowding paridad')
"
```

**Métrica de éxito.** `np.allclose` con 20 puntos 3D. `git commit -m "refactor(D.2): add algorithms/pareto/diversity.py"`.

---

### ETAPA D.3 — crear `algorithms/pareto/hypervolume.py`

**Objetivo atómico.** Crear `compute_hypervolume_mixed` **canónico** (de la versión de clase: loop sobre `objective_senses` + flip de `ref_point`), parametrizado por `objective_senses` y `ref_point` opcional. **No** usar la standalone divergente de `visualization.py`.

**Script de pre-verificación.**
```bash
grep -n "def compute_hypervolume_mixed" algorithms/ga/nsga2.py algorithms/qnas/moqnas.py utils/helpers.py utils/visualization.py
```

**Instrucciones de refactorización.** Crear `algorithms/pareto/hypervolume.py` basado en `nsga2.py:223`. Importar `pymoo.indicators.hv.Hypervolume`.

**Documentación añadida.** Docstring de módulo; docstring NumPy para `compute_hypervolume_mixed(front_raw, objective_senses, ref_point=None)` (Parameters, Returns, Notes sobre el manejo de maximización y la diferencia con la standalone deprecada).

**Script de post-verificación.**
```bash
python -c "
import numpy as np
from algorithms.pareto.hypervolume import compute_hypervolume_mixed as new_hv
from algorithms.ga import nsga2
np.random.seed(42)
front = np.column_stack([np.random.uniform(0.5,0.95,30),
                         np.random.uniform(0.1,8.0,30),
                         np.random.uniform(5.0,80.0,30)])
senses = ['max','min','min']
inst = nsga2.NSGA2.__new__(nsga2.NSGA2); inst.objective_senses = senses
hv_new = new_hv(front, senses); hv_old = inst.compute_hypervolume_mixed(front)
assert np.isclose(hv_new, hv_old, rtol=1e-12), f'HV divergence: {hv_new} vs {hv_old}'
print(f'OK HV paridad: {hv_new}')
"
```

**Métrica de éxito.** `np.isclose` con `rtol=1e-12` contra la versión de clase. `git commit -m "refactor(D.3): add algorithms/pareto/hypervolume.py"`.

---

### ETAPA D.4 — crear `algorithms/pareto/reference_dirs.py` (solo `simplex_lattice` + `to_minimization`)

**Objetivo atómico.** Crear **únicamente** `simplex_lattice` y `to_minimization` (de `nsga3.py:154`/`:124`; idénticas a moead). **`build_reference_directions` NO se centraliza** porque diverge entre nsga3 (sin poda) y moead (poda aleatoria `np.random.choice`); permanece específico por algoritmo.

**Script de pre-verificación.**
```bash
grep -n "def _simplex_lattice\|def _to_minimization\|def _build_reference_directions" algorithms/ga/nsga3.py algorithms/ga/moead.py
```

**Instrucciones de refactorización.** Copiar `simplex_lattice(M, p)` y `to_minimization(fits, objective_senses)` (sin guion bajo) a `algorithms/pareto/reference_dirs.py`. No copiar `build_reference_directions`.

**Documentación añadida.** Docstring de módulo que **documenta explícitamente** por qué `build_reference_directions` queda fuera (divergencia nsga3/moead). Docstrings NumPy para `simplex_lattice` y `to_minimization`.

**Script de post-verificación.**
```bash
python -c "
import numpy as np
from algorithms.pareto.reference_dirs import simplex_lattice as new_sl, to_minimization as new_tm
from algorithms.ga import nsga3, moead
sl_new = new_sl(M=3, p=12)
sl_nsga3 = nsga3.NSGA3._simplex_lattice(None, M=3, p=12)
sl_moead = moead.MOEAD._simplex_lattice(None, M=3, p=12)
assert np.allclose(sl_new, sl_nsga3) and np.allclose(sl_new, sl_moead), 'divergencia simplex'
np.random.seed(3)
fits = np.random.rand(15, 3); senses = ['max','min','min']
tm_new = new_tm(fits, senses)
tm_old = nsga3.NSGA3._to_minimization(None, fits)   # nsga3 usa self.objective_senses
# Para paridad, set senses en una instancia:
inst = nsga3.NSGA3.__new__(nsga3.NSGA3); inst.objective_senses = senses
assert np.allclose(tm_new, inst._to_minimization(fits)), 'divergencia to_minimization'
print('OK reference_dirs paridad (simplex + to_minimization)')
"
```

**Métrica de éxito.** Paridad exacta de `simplex_lattice` entre nueva/nsga3/moead y de `to_minimization`. `build_reference_directions` documentado como no-consolidado. `git commit -m "refactor(D.4): add pareto.reference_dirs (simplex_lattice, to_minimization)"`.

---

### ETAPA D.5 — completar `algorithms/pareto/__init__.py`

**Objetivo atómico.** Exponer la API pública del paquete `pareto` (6 funciones; **sin** `build_reference_directions`).

**Script de pre-verificación.**
```bash
cat algorithms/pareto/__init__.py | wc -l
```

**Instrucciones de refactorización.**
```python
from .dominance      import dominates, fast_nondominated_sort
from .diversity      import crowding_distance
from .hypervolume    import compute_hypervolume_mixed
from .reference_dirs import simplex_lattice, to_minimization
```

**Documentación añadida.** Docstring de paquete enumerando la API y la exclusión de `build_reference_directions`.

**Script de post-verificación.**
```bash
python -c "from algorithms.pareto import (dominates, fast_nondominated_sort, crowding_distance, compute_hypervolume_mixed, simplex_lattice, to_minimization); print('OK')"
```

**Métrica de éxito.** Los 6 símbolos importables desde el paquete. `git commit -m "refactor(D.5): expose algorithms/pareto public API"`.

---

### ETAPA D.6 — migrar `nsga2.py` y borrar duplicados

**Objetivo atómico.** En `nsga2.py`, eliminar los métodos públicos `dominates`, `fast_nondominated_sort`, `crowding_distance`, `compute_hypervolume_mixed` y usar `algorithms.pareto`. **Sustituir** además la standalone divergente: el llamador offline de visualización pasa a usar `algorithms.pareto.compute_hypervolume_mixed` con `objective_senses` explícito.

**Script de pre-verificación.**
```bash
grep -n "def dominates\|def fast_nondominated_sort\|def crowding_distance\|def compute_hypervolume_mixed" algorithms/ga/nsga2.py
grep -i "hypervolume" .refactor_baseline/baseline_nsga2.log | tail -1
```

**Instrucciones de refactorización.**
1. `from algorithms.pareto import dominates, fast_nondominated_sort, crowding_distance, compute_hypervolume_mixed`.
2. Eliminar las 4 definiciones públicas.
3. Reemplazar `self.dominates(a, b)` → `dominates(a, b, self.objective_senses)`; `self.fast_nondominated_sort(f)` → `fast_nondominated_sort(f, self.objective_senses)`; `self.crowding_distance(f, fr)` → `crowding_distance(f, fr)`; `self.compute_hypervolume_mixed(fr)` → `compute_hypervolume_mixed(fr, self.objective_senses)`.

**Script de post-verificación.**
```bash
grep -n "def dominates\|def fast_nondominated_sort\|def crowding_distance\|def compute_hypervolume_mixed" algorithms/ga/nsga2.py \
  && echo "FAIL, quedan duplicados" || echo "OK, duplicados eliminados"
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --seed "$REFACTOR_SEED" --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_D6 --data_path "$DATA_PATH" --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO 2>&1 | tee /tmp/test_D6.log
# Tolerancia (no bit-exacto por caveat cuDNN):
python -c "
import re
def hv(p):
  v=[float(x) for x in re.findall(r'[-+]?\d*\.\d+|\d+', open(p).read().lower().split('hypervolume')[-1][:40])]
  return v[0] if v else None
a=hv('.refactor_baseline/baseline_nsga2.log'); b=hv('/tmp/test_D6.log')
import math; assert a and b and abs(a-b) <= 1e-6 + 1e-3*abs(a), f'{a} vs {b}'; print('OK HV dentro de tolerancia')
"
```

**Métrica de éxito.** Sin duplicados; HV final dentro de tolerancia del baseline seeded; la corrección exacta ya quedó garantizada por las paridades sintéticas D.1–D.3. `git commit -m "refactor(D.6): nsga2 uses algorithms/pareto, remove duplicates"`.

---

### ETAPA D.7 — migrar `moqnas.py` y borrar duplicados

**Objetivo atómico.** Igual que D.6 sobre `moqnas.py`. **Verificación adicional:** la lógica de élite ponderada por `1/rank` vive fuera de los operadores (en la actualización cuántica) y no se toca; solo cambia el origen de los ranks (`fast_nondominated_sort`).

**Script de pre-verificación.**
```bash
grep -n "1/rank\|1 / rank\|elite_weight\|rank_weight" algorithms/qnas/moqnas.py
grep -n "def dominates\|def fast_nondominated_sort\|def crowding_distance\|def compute_hypervolume_mixed" algorithms/qnas/moqnas.py
grep -i "hypervolume" .refactor_baseline/baseline_moqnas.log | tail -1
```

**Instrucciones de refactorización.** Idénticas a D.6 sobre `moqnas.py` (atributo `self.objective_senses` confirmado en `moqnas.py:228`). No tocar la actualización cuántica `1/rank`.

**Script de post-verificación.**
```bash
grep -n "def dominates\|def fast_nondominated_sort\|def crowding_distance\|def compute_hypervolume_mixed" algorithms/qnas/moqnas.py \
  && echo "FAIL" || echo "OK"
python run_all_evolution.py --algo moqnas --num_generations 1 --population_size 2 \
  --seed "$REFACTOR_SEED" --config_file experiment_configs/cifar_mo/config0.yaml \
  --experiment_path /tmp/test_D7 --data_path "$DATA_PATH" --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO 2>&1 | tee /tmp/test_D7.log
diff <(grep -E "rank|elite" .refactor_baseline/baseline_moqnas.log | head -20) \
     <(grep -E "rank|elite" /tmp/test_D7.log | head -20) && echo "OK ranks/elite coinciden" \
  || echo "AVISO: revisar tolerancia en ranks/elite"
```

**Métrica de éxito.** Sin duplicados; HV dentro de tolerancia; logs de rank/élite de la primera generación coinciden. `git commit -m "refactor(D.7): moqnas uses algorithms/pareto, remove duplicates"`.

---

### ETAPA D.8 — migrar `nsga3.py` (solo `simplex_lattice` + `to_minimization`)

**Objetivo atómico.** Eliminar `_simplex_lattice` y `_to_minimization` de `nsga3.py` y usar `algorithms.pareto`. **`_build_reference_directions` se conserva** en `nsga3.py` (variante sin poda; puede internamente llamar a `algorithms.pareto.simplex_lattice`).

**Script de pre-verificación.**
```bash
grep -n "def _simplex_lattice\|def _to_minimization\|def _build_reference_directions" algorithms/ga/nsga3.py
grep -i "hypervolume" .refactor_baseline/baseline_nsga3.log | tail -1
```

**Instrucciones de refactorización.**
1. `from algorithms.pareto import simplex_lattice, to_minimization`.
2. Eliminar `_simplex_lattice` y `_to_minimization`.
3. Reemplazar `self._simplex_lattice(M, p)` → `simplex_lattice(M, p)` (incluida la llamada dentro de `_build_reference_directions`); `self._to_minimization(fits)` → `to_minimization(fits, self.objective_senses)`.
4. `_build_reference_directions` permanece como método (sin poda); solo cambia su dependencia interna de `simplex_lattice`.

**Documentación añadida.** Actualizar el docstring de `_build_reference_directions` para referenciar `pareto.simplex_lattice` y reiterar la política de no-poda.

**Script de post-verificación.**
```bash
grep -n "def _simplex_lattice\|def _to_minimization" algorithms/ga/nsga3.py && echo "FAIL" || echo "OK"
grep -n "def _build_reference_directions" algorithms/ga/nsga3.py && echo "OK, build_reference_directions conservado"
python run_all_evolution.py --algo nsga3 --num_generations 1 --population_size 2 \
  --seed "$REFACTOR_SEED" --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_D8 --data_path "$DATA_PATH" --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO 2>&1 | tee /tmp/test_D8.log
diff <(grep -i hypervolume .refactor_baseline/baseline_nsga3.log | tail -1) \
     <(grep -i hypervolume /tmp/test_D8.log | tail -1) || echo "AVISO: revisar tolerancia"
```

**Métrica de éxito.** `_simplex_lattice`/`_to_minimization` eliminados, `_build_reference_directions` intacto (sin poda), HV dentro de tolerancia. `git commit -m "refactor(D.8): nsga3 uses pareto.simplex_lattice/to_minimization"`.

---

### ETAPA D.9 — migrar `moead.py` (solo `simplex_lattice` + `to_minimization`)

**Objetivo atómico.** Igual que D.8 sobre `moead.py`. **Crítico:** `_build_reference_directions` de moead conserva su **poda aleatoria** (`np.random.choice`); no se reemplaza por la de nsga3.

**Script de pre-verificación.**
```bash
grep -n "def _simplex_lattice\|def _to_minimization\|def _build_reference_directions\|np.random.choice" algorithms/ga/moead.py
grep -i "hypervolume" .refactor_baseline/baseline_moead.log | tail -1
```

**Instrucciones de refactorización.** Idénticas a D.8 sobre `moead.py`. `_build_reference_directions` (con `np.random.choice`) **no se toca** salvo cambiar su llamada interna `self._simplex_lattice` → `simplex_lattice`.

**Script de post-verificación.**
```bash
grep -n "def _simplex_lattice\|def _to_minimization" algorithms/ga/moead.py && echo "FAIL" || echo "OK"
grep -n "np.random.choice" algorithms/ga/moead.py && echo "OK, poda aleatoria preservada"
python run_all_evolution.py --algo moead --num_generations 1 --population_size 2 \
  --seed "$REFACTOR_SEED" --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_D9 --data_path "$DATA_PATH" --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO 2>&1 | tee /tmp/test_D9.log
diff <(grep -i hypervolume .refactor_baseline/baseline_moead.log | tail -1) \
     <(grep -i hypervolume /tmp/test_D9.log | tail -1) || echo "AVISO: revisar tolerancia"
```

**Métrica de éxito.** `_simplex_lattice`/`_to_minimization` eliminados, poda aleatoria preservada, HV dentro de tolerancia. `git commit -m "refactor(D.9): moead uses pareto.simplex_lattice/to_minimization"`.

---

## Bloque E — romper ciclo `core ↔ algorithms`

### ETAPA E.1 — diagnóstico de uso real *(ya ejecutado en Fase 1; solo lectura)*

**Resultado confirmado (sin commit):**
- **`core/config.py:15` → `QChromosomeNetwork`/`QChromosomeParams`: import muerto (Caso 1).** Solo aparece en la línea del import; nunca en el cuerpo. → E.2 = borrado trivial.
- **`base_ga.py:7-8` → `cfg.*`/`evaluation.*`: uso real (Caso 3).** `cfg.ConfigParameters` (`:670`) y `evaluation.EvalPopulation` (`:683`) dentro de la función de orquestación a nivel de módulo. → E.3 = mover wiring a `run_all_evolution.py`. Riesgo alto.

**Instrucciones.** Volcar el diagnóstico a `.refactor_baseline/E1_diagnostico.md` con los outputs:
```bash
{ echo "## E.1 diagnóstico"
  echo "### core/config.py QChromosome:"; grep -n "QChromosome" core/config.py
  echo "### base_ga.py cfg/evaluation:"; grep -n "cfg\.\|evaluation\." algorithms/ga/base_ga.py
  echo "### core importa base_ga?:"; grep -rn "base_ga" core/ || echo "(ninguno)"
} > .refactor_baseline/E1_diagnostico.md
cat .refactor_baseline/E1_diagnostico.md
```

**Métrica de éxito.** `E1_diagnostico.md` registra Caso 1 (E.2) y Caso 3 (E.3). No hay commit.

---

### ETAPA E.2 — romper `core/config.py → algorithms` (Caso 1: borrado trivial)

**Objetivo atómico.** Eliminar `core/config.py:15` (`from algorithms.qnas.chromosome import ...`), import sin uso.

**Script de pre-verificación.**
```bash
grep -n "QChromosome" core/config.py   # debe aparecer solo la línea 15
python -c "import sys; from core import config; print([m for m in sys.modules if m.startswith('algorithms')])"
```

**Instrucciones de refactorización.** Borrar la línea 15. (Confirmado que `QChromosome*` no se referencia en ningún método.)

**Script de post-verificación.**
```bash
python -c "
import sys
from core import config
mods = [m for m in sys.modules if m.startswith('algorithms')]
assert not mods, f'core.config sigue cargando: {mods}'
print('OK, core.config no importa algorithms')
"
python run_all_evolution.py --algo moqnas --num_generations 1 --population_size 2 \
  --seed "$REFACTOR_SEED" --config_file experiment_configs/cifar_mo/config0.yaml \
  --experiment_path /tmp/test_E2 --data_path "$DATA_PATH" --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** `core.config` no carga `algorithms`; smoke run moqnas OK. `git commit -m "refactor(E.2): remove dead algorithms import from core.config"`.

---

### ETAPA E.3 — romper `algorithms/ga/base_ga.py → core` (Caso 3: mover wiring) — **riesgo alto**

**Objetivo atómico.** Eliminar `from core import evaluation` y `from core import config as cfg` (`base_ga.py:7-8`) moviendo el wiring (`cfg.ConfigParameters`, `evaluation.EvalPopulation`) hacia `run_all_evolution.py`, e inyectando `config`/`eval_pop` por constructor/parámetros de la función de orquestación.

> **Nota de diseño.** El ciclo no es dead code: la función de orquestación de `base_ga.py` construye config y eval_pop. Romper el ciclo implica un mini-rediseño del entry point. Si el wiring no puede extraerse limpiamente, documentar el límite y dejar el import como dependencia conocida en vez de forzar una abstracción frágil.

**Script de pre-verificación.**
```bash
python -c "
import sys
from algorithms.ga import base_ga
print('core cargado por base_ga:', [m for m in sys.modules if m.startswith('core')])
"
sed -n '650,700p' algorithms/ga/base_ga.py   # revisar la función de orquestación completa
```

**Instrucciones de refactorización.** Mover la construcción de `config = cfg.ConfigParameters(...)` y `eval_pop = evaluation.EvalPopulation(...)` a `run_all_evolution.py`; pasar `config`/`eval_pop` ya construidos a la función/clase de `base_ga`. Eliminar los imports de core. Verificar que ningún otro punto de `base_ga.py` use `cfg.`/`evaluation.`.

**Documentación añadida.** Actualizar el docstring de la función de orquestación para reflejar inyección de dependencias; docstring del nuevo punto de wiring en `run_all_evolution.py`.

**Script de post-verificación.**
```bash
python -c "
import sys
for m in list(sys.modules):
    if 'algorithms' in m or m.startswith('core'): del sys.modules[m]
from algorithms.ga import base_ga
mods = [m for m in sys.modules if m.startswith('core')]
assert not mods, f'base_ga sigue cargando: {mods}'
print('OK, base_ga no importa core')
"
for algo in nsga2 nsga3 moead; do
  python run_all_evolution.py --algo $algo --num_generations 1 --population_size 2 \
    --seed "$REFACTOR_SEED" --config_file experiment_configs/cifar/config0.yaml \
    --experiment_path /tmp/test_E3_$algo --data_path "$DATA_PATH" --dataset cifar10 \
    --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO || exit 1
done
echo "OK 3 GA"
```

**Métrica de éxito.** `base_ga` no importa `core`; los 3 GA terminan con frente no vacío. Si el wiring no se puede extraer, documentar el límite y revertir a un import explícito acotado. `git commit -m "refactor(E.3): move base_ga wiring to entry point, break cycle"`.

---

## Bloque F — limpieza final

### ETAPA F.1 — consolidar `_Wrap` en `dataset_utils/factory.py`

**Objetivo atómico.** Mover `_Wrap` (definida en `factory.py:110` y `:215`) a nivel de módulo como `_TransformWrapper`.

**Script de pre-verificación.**
```bash
grep -c "class _Wrap" dataset_utils/factory.py   # debe ser 2
```

**Instrucciones de refactorización.** Definir `_TransformWrapper(Dataset)` antes de `build_datasets`; reemplazar las dos definiciones internas y sus usos (`train_dataset = _Wrap(...)`, etc.) por `_TransformWrapper`.

**Documentación añadida.** Docstring NumPy para `_TransformWrapper` (Parameters: subset, transform; comportamiento de `__getitem__`).

**Script de post-verificación.**
```bash
grep -c "class _Wrap" dataset_utils/factory.py        # debe ser 0
grep -c "class _TransformWrapper" dataset_utils/factory.py  # debe ser 1
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --seed "$REFACTOR_SEED" --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_F1 --data_path "$DATA_PATH" --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** Una sola definición; smoke run OK (rama torchvision). `git commit -m "refactor(F.1): consolidate _Wrap into _TransformWrapper"`.

---

### ETAPA F.2 — eliminar `load_evolved_data` standalone

**Objetivo atómico.** Confirmar que la versión incompleta (antes en `helpers.py:430`, no portada en C.1) no existe en `utils/*` y que nadie la importa. Canónica = `core/config.ConfigParameters.load_evolved_data`.

**Script de pre-verificación.**
```bash
grep -rn "from utils.helpers import.*load_evolved_data\|from utils.io import.*load_evolved_data\|helpers.load_evolved_data\|io.load_evolved_data" . --include="*.py"
```

**Instrucciones de refactorización.** Si el grep no devuelve nada, asegurar que no quedó ninguna `def load_evolved_data` en `utils/*` ni en la facade. Documentar la canónica.

**Script de post-verificación.**
```bash
grep -rn "def load_evolved_data" utils/ && echo "FAIL" || echo "OK, no en utils/"
grep -n "def load_evolved_data" core/config.py && echo "OK, canónica en core.config"
python -c "from core.config import ConfigParameters; print('OK')"
```

**Métrica de éxito.** Cero defs en `utils/`, una en `core/config.py`. `git commit -m "refactor(F.2): remove standalone load_evolved_data"`.

---

### ETAPA F.3 — marcar `utils/helpers.py` como deprecated

**Objetivo atómico.** Añadir `DeprecationWarning` a la facade. No destructivo.

**Script de pre-verificación.**
```bash
wc -l utils/helpers.py
python -W error::DeprecationWarning -c "from utils.helpers import load_yaml" 2>&1 | grep -i deprecat && echo "ya hay warning" || echo "sin warning aún"
```

**Instrucciones de refactorización.** Añadir al tope:
```python
import warnings
warnings.warn(
    "utils.helpers está deprecado. Importa desde utils.io, utils.logging_utils, "
    "utils.experiment, utils.dataset o utils.visualization.",
    DeprecationWarning, stacklevel=2,
)
```

**Script de post-verificación.**
```bash
python -W default::DeprecationWarning -c "from utils.helpers import load_yaml" 2>&1 | grep -i deprecat && echo "OK warning"
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --seed "$REFACTOR_SEED" --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_F3 --data_path "$DATA_PATH" --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO 2>&1 | grep -i deprecat \
  && echo "AVISO: código interno aún usa helpers (C.x incompleto)" || echo "OK, interno migrado"
```

**Métrica de éxito.** Warning al importar desde fuera; el smoke run no emite `DeprecationWarning`. `git commit -m "refactor(F.3): deprecate utils/helpers.py facade"`.

---

### ETAPA F.4 — archivar `old_files/`

**Objetivo atómico.** Confirmar que `old_files/` no se referencia desde código activo (verificado en Fase 1: sin refs) y archivarlo.

**Script de pre-verificación.**
```bash
grep -rn "old_files" run_all_evolution.py retrain_model.py retrain_parallel.py \
  algorithms/ core/ dataset_utils/ utils/ scripts/ --include="*.py" || echo "OK, sin refs"
```

**Instrucciones de refactorización.** Si no hay refs:
```bash
git checkout -b legacy/old_files
git push origin legacy/old_files
git checkout refactor/update-2026-staged
git rm -r old_files/
```

**Script de post-verificación.**
```bash
test ! -d old_files && echo "OK, old_files eliminado"
git ls-remote origin legacy/old_files | grep -q legacy/old_files && echo "OK, backup en rama"
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 2 \
  --seed "$REFACTOR_SEED" --config_file experiment_configs/cifar/config0.yaml \
  --experiment_path /tmp/test_F4 --data_path "$DATA_PATH" --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO
```

**Métrica de éxito.** Carpeta eliminada local, backup en rama remota, smoke run final OK. `git commit -m "refactor(F.4): archive old_files/ to legacy branch"`.

---

## Resumen de etapas

| ID | Bloque | Riesgo | Commit esperado |
|---|---|---|---|
| 0 | Línea base git + import map | nulo | `import map and staging branch` |
| 0.5 | Semillas + baselines seeded | bajo | `add global seeding and reproducible baselines` |
| A.1–A.8 | Settings y renombrados | muy bajo | 8 commits |
| B.1–B.2 | Side-effects de import | bajo | 2 commits |
| C.1–C.6 | Split de helpers | bajo | 6 commits |
| D.1–D.9 | Operadores Pareto | medio | 9 commits con paridad sintética |
| E.1 | Diagnóstico ciclo | nulo | sin commit |
| E.2 | core.config → algorithms (Caso 1) | bajo | 1 commit |
| E.3 | base_ga → core (Caso 3) | **alto** | 1 commit |
| F.1–F.4 | Limpieza | muy bajo | 4 commits |

**Total: 33 commits atómicos** (32 originales + Etapa 0.5). E.1 no produce commit.

## Notas de validación

1. **Exportar `PROJECT_ROOT`, `DATA_PATH`, `REFACTOR_SEED`** antes de cualquier etapa (ver sección de variables de entorno).
2. **Bloque D:** la corrección de operadores se garantiza con las paridades sintéticas deterministas (D.1–D.4); los checks end-to-end usan tolerancia por el caveat cuDNN documentado en 0.5. No exigir bit-exacto end-to-end.
3. **`_build_reference_directions` NO se consolida** (nsga3 sin poda vs moead con `np.random.choice`). Solo se comparten `simplex_lattice` y `to_minimization`.
4. **Etapa E.3 (riesgo alto):** el ciclo `base_ga → core` es uso real; requiere mover wiring al entry point. Si no se puede extraer limpiamente, documentar el límite en vez de forzar una abstracción frágil.
5. **Sin tests automáticos.** Cada smoke run consume GPU; estimación ~34 corridas de 1 generación/población 2.
6. **El notebook `experiment_configs/cifar/config_probs.ipynb`** se preserva en A.6/A.7; verificar sus rutas internas tras los renombrados.
