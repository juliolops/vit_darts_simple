# Documentación de cambios — Mejoramiento y actualización de MoQ-NAS

Este documento describe, de forma autocontenida, **todos los cambios** realizados
sobre el código de MoQ-NAS durante el proceso de mejoramiento y actualización, y
explica los **conceptos** necesarios para analizarlos y revisarlos. Está pensado
para que un revisor que no participó en el proceso pueda entender qué se cambió,
por qué, y cómo se verificó cada cambio.

El trabajo se realizó íntegramente sobre la rama `refactor/update-2026-staged`,
en commits atómicos (un cambio verificable por commit). Se divide en dos grandes
fases:

1. **Refactor estructural** (`roadmap 0 → F.3`): reorganiza el código sin cambiar
   su comportamiento numérico. Objetivo: legibilidad, modularidad y eliminación
   de deuda técnica.
2. **Mejoras funcionales** (`IMPROVEMENT_STRATEGY`, 6 áreas): añade capacidades
   nuevas (objetivos configurables, BF16, caché, checkpoint/resume,
   paralelización, lanzador de experimentos).

---

## Índice

- [1. Conceptos transversales](#1-conceptos-transversales)
- [2. Fase I — Refactor estructural (roadmap 0 → F.3)](#2-fase-i--refactor-estructural-roadmap-0--f3)
- [3. Correcciones posteriores al roadmap](#3-correcciones-posteriores-al-roadmap)
- [4. Fase II — Mejoras funcionales (6 áreas)](#4-fase-ii--mejoras-funcionales-6-áreas)
- [5. Conceptos clave explicados en profundidad](#5-conceptos-clave-explicados-en-profundidad)
- [6. Cómo verificar y reproducir](#6-cómo-verificar-y-reproducir)
- [7. Apéndice: tabla completa de commits](#7-apéndice-tabla-completa-de-commits)

---

## 1. Conceptos transversales

Tres ideas atraviesan todo el trabajo y conviene entenderlas antes de revisar los
cambios.

### 1.1 Determinismo y *seeding* por candidato

NAS evalúa **poblaciones** de arquitecturas. Si entrenar la misma arquitectura,
con la misma semilla, produce resultados distintos según en qué *worker* (proceso)
o en qué orden se entrenó, entonces la "señal de búsqueda" se contamina con ruido
y los experimentos no son reproducibles ni comparables.

La solución implementada (Etapa 0.6) es **sembrar las RNG por candidato**: antes de
entrenar cada arquitectura se reinicializan `random`, `numpy` y `torch` con una
semilla derivada de `(semilla_global, generación, candidate_id)`. Como esa semilla
no depende del *worker* ni del orden, el resultado de un candidato es una **función
pura** de su identidad. Función relevante: `utils/seeding.py::seed_candidate`,
invocada en `core/evaluation.py::run_individuals`.

**Consecuencia práctica:** `best_accuracy` y `total_params` de cada candidato son
**bit-exactos** entre corridas con la misma semilla, sin importar el número de
*threads* ni las GPUs usadas. Esta propiedad es la base de casi toda la verificación.

### 1.2 Verificación bit-exacta como criterio de aceptación

Cada cambio del refactor se valida demostrando que **no altera la numérica**: se
compara la salida por candidato contra una **referencia** generada al inicio del
proceso (`.refactor_baseline/expB_run1.log` para nsga2, `expC_moqnas.log` para
moqnas). Si el diff es vacío, el cambio es seguro.

Para operadores matemáticos (Bloque D) se usan además **scripts de paridad
sintética**: se ejecuta el operador nuevo y el viejo sobre los mismos datos
aleatorios (con semilla local) y se exige igualdad exacta.

### 1.3 Reproducibilidad de objetivos: por qué el tiempo no, y FLOPs sí

El objetivo `cuda_inference_time` es **tiempo de pared medido**: depende de la carga
de la GPU en ese instante, así que **no es reproducible** entre corridas (varía
decenas a cientos de ms). En cambio `total_flops` y `total_params` son
**deterministas** (se cuentan, no se miden).

**Implicación:** cuando el conjunto de objetivos incluye el tiempo, el frente de
Pareto resultante no es reproducible aunque la accuracy/params sí lo sean. Por eso,
para pruebas que exigen reproducibilidad de la búsqueda completa (p. ej. el test de
*resume* del Área 6) se usa un config con objetivos basados en FLOPs
(`experiment_configs/cifar_mo/config0_3_flops.yaml`), con el cual una corrida
entera es bit-reproducible.

---

## 2. Fase I — Refactor estructural (roadmap 0 → F.3)

El refactor se ejecutó en bloques. Cada etapa es un commit con verificación propia.

### Etapas 0, 0.5 y 0.6 — línea base y reproducibilidad

- **0** (`d36057b`): mapa de imports y rama de trabajo. Red de seguridad antes de
  tocar nada.
- **0.5** (`a12925f`): `utils/seeding.py::set_global_seeds` siembra todas las RNG
  globales; se añade el flag `--seed`. Se generan los *baselines* seedeados.
- **0.6** (`91faf92`): **hallazgo clave del proceso.** Las corridas seedeadas aún
  daban accuracy distinta. La causa raíz era que `core/cnn/input.py`
  (`GenericDataLoader.__init__`) **resembraba las RNG globales con `int(time())`**,
  destruyendo `--seed`. Se eliminó ese *reseed*, se añadió `seed_candidate` (sección
  1.1) y se reinicializa el generador del `DataLoader` por candidato. Además,
  `--num_generations` ahora sí sobreescribe `max_generations` para qnas/moqnas
  (antes el flag se ignoraba). Resultado verificado: accuracy por candidato
  bit-exacta entre corridas e independiente del número de *threads*.

### Bloque A — `settings.py` y renombrados (`A.1`–`A.8`)

Centraliza rutas/constantes y normaliza nombres:

- `settings.py` con `CFG_OBJ_PATH` y `TRAIN_TIMEOUT`; sus consumidores (`nsga2`,
  `moqnas`, `trainer`) migrados a leerlos de ahí (`A.1`–`A.4`).
- Renombrados de carpetas y extensiones (`A.5`–`A.7`):
  `configs/` → `dataset_configs/` (metadata de datasets),
  `config_files/` → `experiment_configs/` (+ subcarpetas `cifar`, `cifar_mo`, …),
  y las extensiones `.txt` → `.yaml`.
- Scripts `.sh` actualizados a las rutas nuevas (`A.8`).

### Bloque B — eliminar *side-effects* en import (`B.1`, `B.2`)

`core/cnn/master.py` y `core/cnn/trainer.py` **creaban el directorio `logs/` al ser
importados** (efecto colateral en tiempo de import). Se reemplazó por
`logging.getLogger(__name__)`; la creación del directorio se difiere a la instancia
del *trainer*. Importar esos módulos ya no tiene efectos colaterales.

### Bloque C — dividir `utils/helpers.py` (`C.1`–`C.6`)

`helpers.py` era un archivo monolítico de ~1200 líneas. Se dividió en cinco módulos
temáticos sin romper a ningún consumidor:

| Módulo nuevo | Contenido |
|---|---|
| `utils/io.py` | carga/guardado YAML/JSON/pickle, escrituras atómicas, caché en disco |
| `utils/logging_utils.py` | `init_log` |
| `utils/experiment.py` | `natural_key`, validación de rutas, limpieza de directorios, tiempos |
| `utils/dataset.py` | descarga y metadata de datasets |
| `utils/visualization.py` | gráficos y agregación de resultados |

En `C.6`, `helpers.py` quedó como **fachada** (`facade`) de 24 líneas que reexporta
todos los nombres, de modo que los 14 módulos que hacían `from utils.helpers import …`
siguen funcionando sin cambios. Verificado bit-exacto.

### Bloque D — operadores Pareto compartidos (`D.1`–`D.9`)

Los cuatro algoritmos multi-objetivo (nsga2, moqnas, nsga3, moead) tenían **copias
duplicadas** de los mismos operadores. Se creó el paquete `algorithms/pareto/` con
las versiones canónicas y parametrizadas:

- `dominance.py`: `dominates`, `fast_nondominated_sort`.
- `diversity.py`: `crowding_distance`.
- `hypervolume.py`: `compute_hypervolume_mixed` (la versión **canónica**; la
  *standalone* divergente de `visualization.py` quedó marcada como obsoleta).
- `reference_dirs.py`: `simplex_lattice`, `to_minimization`.

Estrategia (la misma del Bloque C): primero crear el paquete (`D.1`–`D.5`) y
verificar **paridad sintética** contra cada copia; luego migrar cada algoritmo y
borrar sus duplicados (`D.6`–`D.9`), eliminando ~320 líneas. `build_reference_directions`
**no** se consolidó porque diverge legítimamente entre nsga3 (sin poda) y moead
(poda aleatoria) — esto está documentado en el código.

> **Bug latente encontrado y corregido en `D.8`:** al borrar `fast_nondominated_sort`
> de NSGA2 (`D.6`), `nsga3` —que lo heredaba— quedó llamando a un método
> inexistente. Cualquier corrida nsga3 de ≥2 generaciones habría fallado con
> `AttributeError`. El roadmap no lo había previsto; se corrigió recableando esas
> llamadas al paquete.

### Bloque E — romper el ciclo `core ↔ algorithms` (`E.1`–`E.3`)

Había una dependencia circular entre `core/` y `algorithms/`:

- `E.2` (`fd54643`): `core/config.py` importaba `QChromosome*` de `algorithms`
  pero **nunca lo usaba** (import muerto) → eliminado.
- `E.3` (`69c5814`): `algorithms/ga/base_ga.py` importaba `core` para un bloque de
  demostración `if __name__ == "__main__"` obsoleto (rutas pre-A.6); el wiring real
  ya vivía en `run_all_evolution.py`. Se borró el bloque muerto y los imports.

Resultado: importar `core.config` o `algorithms.ga.base_ga` ya no arrastra al otro
paquete; el ciclo quedó roto en ambas direcciones.

### Bloque F — limpieza final (`F.1`–`F.3`)

- `F.1`: se consolidaron dos clases internas `_Wrap` idénticas (en
  `dataset_utils/factory.py`) en una sola `_TransformWrapper` a nivel de módulo
  (además, ahora los datasets son *picklables* para *workers* de tipo `spawn`).
- `F.2`: verificación de que la `load_evolved_data` *standalone* (incompleta) ya no
  existe; la canónica es `core.config.ConfigParameters.load_evolved_data`.
- `F.3`: `utils/helpers.py` (la fachada) emite un `DeprecationWarning` al importarse,
  para guiar el código nuevo hacia los módulos temáticos.

---

## 3. Correcciones posteriores al roadmap

- `021c129` **fix(qnas):** el config mono-objetivo `experiment_configs/cifar/config0_3.yaml`
  no parseaba (la validación exige `mo_crossover_strategy`). Se reparó el config y,
  en `run_all_evolution.py`, el *branch* de qnas ahora filtra los argumentos por la
  firma de `initialize_qnas` (ignora las claves multi-objetivo que no acepta). Este
  `TypeError` **ya existía antes del refactor** (reproducido en el commit de
  partida); no era una regresión.
- `8090140` **fix(gitignore):** el patrón `exp*/` (para directorios de resultados)
  también atrapaba `experiment_configs/` tras el renombrado A.6; se añadió la
  excepción.
- `52d099f` **chore:** se eliminó `algorithms/qnas/qnas.py` (código muerto; el módulo
  vivo es `qnas2.py`, importado como `qnas2 as qnas`).

---

## 4. Fase II — Mejoras funcionales (6 áreas)

Las mejoras se diseñaron primero en `IMPROVEMENT_STRATEGY.md` (auditoría de 6 áreas)
y luego se implementaron en el orden recomendado. Un principio rector: **identidad de
evaluación compartida** — caché, precisión y checkpoint usan la misma noción de "qué
hace única a una evaluación" (`core/eval_cache.py::compute_fingerprint`).

### Área 3 — Objetivos configurables y FLOPs (`074af91`, `db34eed`, `6c3f691`, `a87dd94`)

**Objetivo:** permitir cualquier combinación de objetivos (p. ej. accuracy +
parámetros + FLOPs, o accuracy + FLOPs) sin hardcodear ninguna.

La infraestructura ya existía en su mayoría (FLOPs ya se computaba en
`HardwareMetrics`; el paquete `pareto` ya era agnóstico al número de objetivos). El
trabajo fue de **validación y des-hardcodeo**:

- **Validación al parsear** (`core/config.py::_check_objectives`): cada objetivo
  debe resolver a exactamente un sentido en `dataset_configs/cfg_obj.json` y ser
  producido por alguna métrica configurada. Falla temprano con un mensaje claro.
  > Esto cerró un **bug real**: un objetivo sin sentido era un *warning* que se
  > ignoraba, dejando `objective_senses` más corto que la matriz de *fitness* y
  > **volteando las columnas equivocadas** en silencio. También se añadió el sentido
  > faltante de `fairness_score`.
- **Normalizadores genéricos** en `ScalarizedFitness` (sustituyen los hardcodeos de
  `max_params`/`max_inference_time`), con retro-compatibilidad bit-exacta.
- **Configs de ejemplo** con objetivos FLOPs (`config0_3_flops.yaml`,
  `config0_3_acc_flops.yaml`).
- **Gráficos Pareto** dirigidos por los nombres de objetivos (antes asumían
  accuracy/params/tiempo).

**Resultado clave:** con objetivos FLOPs, una corrida completa es bit-reproducible
(frente e hipervolumen idénticos entre corridas) — habilita el test de aceptación
del Área 6.

### Área 4 — Migración de precisión FP16 → BF16 (`72fcdf6`, `bc6f25a`, `edaa0b1`)

**Concepto:** FP16 tiene 5 bits de exponente; BF16 tiene 8 (igual rango dinámico
que FP32). En NAS, donde se entrenan arquitecturas heterogéneas con presupuestos
cortos, FP16 sufre *underflow* de gradientes que inyecta ruido en los *scores*. BF16
elimina ese fallo sin necesidad de `GradScaler`. En hardware Ada (L40S) ambos
formatos tienen el mismo rendimiento.

Cambios:

- **`core/precision.py`**: punto único que resuelve `train.precision`
  (`fp32 | fp16 | bf16`). El flag legacy `mixed_precision: true` se mapea a `fp16`,
  así los configs antiguos siguen siendo reproducibles.
- **`trainer.py`**: el `dtype` del `autocast` sale de la política (antes estaba
  hardcodeado a `float16`); el `GradScaler` se construye **solo** para `fp16`;
  `bf16`/`fp32` usan un *backward* sin escalado. Si se pide `bf16` en hardware sin
  soporte nativo, se lanza un `RuntimeError` explícito (nunca *fallback* silencioso).
- **`fairness.py`**: la evaluación de *fairness* ahora sigue la misma política (antes
  auto-seleccionaba BF16, lo que era inconsistente con un entrenamiento en FP16).

Verificado: `fp16` bit-exacto vs la referencia; `bf16` determinista en la L40S.
La precisión es parte del *fingerprint* de la caché (Área 1) y del *checkpoint*
(Área 6): una entrada FP16 nunca sirve para una corrida BF16.

### Área 1 — Caché de evaluación unificada (`a42a7af`, `e87428f`)

**Problema:** existían **tres** cachés por algoritmo (`base_ga`, `nsga2`, `qnas2`),
con semánticas distintas, y todas con una clave defectuosa: usaban **solo el
cromosoma de red**, ignorando los hiperparámetros continuos evolucionados (p. ej.
`backbone_percentage`) y la configuración de entrenamiento.

Solución: **`core/eval_cache.py::CachedEvaluator`**, un *wrapper* sobre
`EvalPopulation` (la frontera que todos los algoritmos comparten) con el mismo
contrato `__call__`. Detalles:

- **Clave compuesta:** `(red, hiperparámetros, fingerprint)`. El *fingerprint* es un
  hash de los campos relevantes de la config (dataset, presupuesto, optimizador,
  *batch*, **precisión**, **objetivos**, semilla). Una caché copiada entre
  experimentos con config distinta se **ignora**, no se reutiliza por error.
- **Almacenamiento:** un pickle por experimento, escrito atómicamente.
- Activación con el flag existente `--use_cache`; cubre **todos** los algoritmos
  (moqnas incluido, que antes no tenía caché).

Las tres cachés legacy se eliminaron (-183 líneas) con paridad bit-exacta. En una
corrida ya cacheada, la segunda ejecución hace 0 entrenamientos (4 s frente a ~90 s).

### Área 6 — Checkpoint y reanudación (`dff9606`)

**Concepto:** MO-QNAS es un algoritmo cuántico-evolutivo. Su estado en una frontera
de generación incluye dos poblaciones acopladas que deben restaurarse juntas:

- **Población cuántica (`Qpop`)**: un tensor de PMFs (distribuciones de probabilidad)
  que codifican la distribución de búsqueda aprendida — acumula actualizaciones de
  todas las generaciones previas.
- **Estado élite acoplado**: la EMA de élites (`_q_ema` en `update_strategies.py`) y
  el contador de actualizaciones. Restaurar `Qpop` sin la EMA haría que la primera
  actualización tras reanudar parta de un prior equivocado, corrompiendo la búsqueda.
- **Archivo de Pareto externo** y el contador de generación.

`algorithms/qnas/checkpoint.py` guarda **atómicamente** todo ese estado en cada
frontera de generación (al final de `go_next_gen`, tras la actualización cuántica y
antes de avanzar el contador), incluyendo **todas las RNG** (numpy, python, torch
CPU/CUDA). Detalles de diseño:

- **`--resume` explícito**: sin el flag, un checkpoint existente se **ignora** y la
  corrida empieza en gen 0 (comportamiento seguro por defecto). En esta etapa solo
  moqnas lo soportaba; **más tarde se extendió a toda la familia GA** (ver la sección
  "Extensión post-estrategia" más abajo).
- **Validación de config al reanudar**: se compara campo a campo un bloque de config
  (incluye `max_generations`, porque gobierna los *schedules* del *update* cuántico,
  además de objetivos, precisión y semilla). Cualquier diferencia aborta nombrando
  el campo.
- **Detalle fino encontrado:** un `torch.Tensor` crudo se serializa con metadata no
  determinista (mismo contenido, distintos bytes). Las RNG de torch se guardan como
  arrays de numpy para que dos checkpoints idénticos sean byte-idénticos.

**Test de aceptación (el más fuerte del proceso):** una corrida de 4 generaciones
**interrumpida con SIGKILL** tras la generación 2 y **reanudada** produce un estado
**bit-exacto** (PMFs, EMA, archivo, hipervolumen, estado RNG) frente a la corrida
ininterrumpida. Se usa el config FLOPs (sección 1.3) para que la comparación sea
posible.

### Área 5 — Paralelización del entrenamiento (`5010fdf`, `1943665`)

El esquema previo **pre-particionaba** la población entre *workers* de forma estática
(round-robin), dejando ociosos a los que sacaban arquitecturas rápidas mientras otros
seguían con arquitecturas profundas (*stragglers*).

- **Scheduler *work-stealing*** (`core/evaluation.py`): una única cola de tareas que
  N *workers* consumen hasta vaciarla. Como el *seeding* es por `candidate_id`, el
  determinismo se preserva sin importar qué *worker* toma qué tarea. Se añadió
  instrumentación del tiempo ocioso por *worker*.
- **Reintento ante OOM**: un *out-of-memory* de CUDA dispara `empty_cache()` y un
  reintento antes de puntuar el candidato como 0.0 con un *log* de error. Así la
  presión de memoria transitoria no inyecta candidatos de *fitness* cero en la
  búsqueda de forma silenciosa.

Verificado bit-exacto (incluso **entre topologías**: 1 GPU vs 2 GPUs dan el mismo
resultado por candidato). **Hallazgo honesto:** a escala pequeña (8 candidatos
pequeños) el *speedup* multi-GPU no se materializa porque el cuello de botella es el
costo fijo por *worker* (construir el `DataLoader`), no el cómputo de GPU; el
*speedup* aparece con cargas que saturen la GPU.

### Área 2 — Lanzador de experimentos (`18e0857`, `523a063`)

**Problema:** había una proliferación de scripts `.sh` casi idénticos; además, los
*repeats* nunca pasaban `--seed`, así que solo diferían por ruido de *timing*.

**`launch.py`** expande una **matriz YAML** (en `experiment_matrices/`) en una
invocación de `run_all_evolution.py` por celda (experimento × repeat):

- Reparte las celdas sobre un pool de *slots* de GPU (una corrida por *slot*; un
  fallo no detiene a las demás).
- Da **semilla explícita por repeat** (`seed_base + i`).
- Registra el comando exacto y `CUDA_VISIBLE_DEVICES` en cada directorio.
- Soporta `--dry-run` (muestra los comandos sin ejecutar).

Los scripts `run_ea_1.sh`, `run_moqnas_1.sh`, `run_qnas_1.sh` pasaron a ser
*wrappers* de una línea sobre el lanzador. Verificado: dos celdas concurrentes en 2
GPUs, semillas por repeat, y `repeat_1` (semilla 42) bit-exacto vs la referencia.

### Extensión post-estrategia — Checkpoint/resume para la familia GA (`a7b966f`, `a0e6216`, `d2c9dcb`)

El Área 6 dejó el checkpoint solo para MO-QNAS. Una corrida de la familia GA
(`GA`, `NSGA-II`, `NSGA-III`, `MOEA/D`) que perdiera energía en la generación 80 de
100 debía reiniciarse desde 0. Esta extensión lo resuelve, reusando la
infraestructura del Área 6 en vez de duplicarla. La estrategia previa se documentó
en `ESTRATEGIA_CHECKPOINT_GA.md` (commit `a24e5a2`).

**Generalización del módulo (`a7b966f`).** `algorithms/qnas/checkpoint.py` se movió a
`algorithms/checkpoint.py` y se hizo **agnóstico al motor**: el módulo maneja solo lo
transversal (contador de generación, *bookkeeping* común, captura de RNG, escritura
atómica, validación del bloque de config, flag `_resumed`) y delega el estado
específico a tres métodos que cada motor implementa:

- `_checkpoint_config_block()` — config que define la identidad (validada al reanudar);
- `_checkpoint_state()` — el estado restaurable del motor;
- `_restore_state(state)` — lo restaura.

Esto mapea la jerarquía de clases `GA → NSGA2 → {NSGA3, MOEAD}`: cada subclase
**extiende** los métodos del padre con `super()`. MO-QNAS implementa los suyos
reproduciendo su contenido previo (regresión cero: dos corridas seedeadas siguen
dando checkpoints byte-idénticos).

**Estado por algoritmo (`a0e6216`).** Lo que cada motor declara como restaurable:

| Algoritmo | Estado propio (además del común) |
|---|---|
| `GA` | `population`, **`pop_params`** (genes continuos evolucionados), `fitnesses` |
| `NSGA-II` | `population_ids`, archivo de Pareto (`pareto_global_*`), `fronts_history` |
| `NSGA-III` | `_ref_dirs` (direcciones de referencia) + `ref_divisions` en el config block |
| `MOEA/D` | **`z`** (punto ideal acumulado a lo largo de toda la búsqueda), `weights`, `neighbors` |

`save_checkpoint` se engancha al final de cada `go_next_gen` (la frontera de
generación). Cada `evolve` gana una rama `_resumed` que **salta la inicialización de
la gen 0** y continúa en `g+1`. En NSGA-II hubo una sutileza: la población padre vive
en variables locales del bucle (`pop_old`/`fits_old`/`ids_old`), así que la rama de
*resume* las **reconstruye** desde los atributos restaurados antes de entrar al
`while`. En MOEA/D, `z` se restaura **después** de que `initialize_ga` lo recalcula
desde cero (el `z` restaurado es el autoritativo).

> **Los dos atributos de mayor riesgo** —`pop_params` (GA) y `z` (MOEA/D)— son
> precisamente los que un checkpoint ingenuo olvidaría: el primero son los
> hiperparámetros continuos evolucionados; el segundo es estado acumulado entre
> generaciones (análogo a la `_q_ema` de MO-QNAS). El test de aceptación los cubre.

**Reanudación de un *batch* completo (`d2c9dcb`).** El lanzador (`launch.py`) acepta
ahora una clave de matriz `resume: true` **o** un flag `--resume` (que tiene
prioridad), que añade `--resume` a cada celda. Relanzar un *batch* interrumpido es
rerunear la misma matriz con `--resume`; cada celda retoma desde su propio
`checkpoint.pkl`.

**Test de aceptación (`.refactor_baseline/ga_checkpoint_check.sh`, 18/18 PASS).** Para
**cada uno de los 4 algoritmos**: una corrida de 4 generaciones **interrumpida con
SIGKILL tras la gen 2** y **reanudada** produce un estado **bit-exacto** (población,
`pop_params`, archivo de Pareto, `z`, `_ref_dirs`, estado RNG de numpy) frente a la
corrida ininterrumpida. Más los *guard checks*: sin `--resume` arranca en gen 0, y un
*mismatch* de config (p. ej. `num_generations` distinto) aborta nombrando el campo.
Se usan objetivos FLOPs para nsga2/nsga3/moead y `best_accuracy` (determinista) para
GA, de modo que la corrida completa sea reproducible (sección 1.3).

---

## 5. Conceptos clave explicados en profundidad

### 5.1 Frente de Pareto, dominancia e hipervolumen

En optimización multi-objetivo no hay un "mejor" único: una solución **domina** a
otra si no es peor en ningún objetivo y es estrictamente mejor en al menos uno. El
**frente de Pareto** es el conjunto de soluciones no dominadas (los mejores
compromisos). El **hipervolumen** mide el "volumen" del espacio objetivo dominado por
el frente respecto a un punto de referencia: es un número único que resume la calidad
de un frente (mayor es mejor). El paquete `algorithms/pareto/` centraliza estos
cálculos; `objective_senses` (lista de `'max'`/`'min'`) le indica cómo convertir cada
objetivo a forma de minimización.

### 5.2 Población cuántica y PMFs (MO-QNAS)

A diferencia de un AG clásico que evoluciona individuos concretos, QNAS evoluciona
**distribuciones de probabilidad** (una por cada nodo de decisión de la arquitectura,
sobre los operadores disponibles). Estas PMFs (*probability mass functions*) son la
"memoria" de la búsqueda: cada generación se **muestrean** arquitecturas concretas a
partir de ellas, se evalúan, y las mejores (élites) **empujan** las PMFs hacia las
zonas prometedoras (la "actualización cuántica"). Por eso el checkpoint debe guardar
las PMFs **y** el estado de élites: son las dos mitades del mismo mecanismo.

### 5.3 Identidad de evaluación (*fingerprint*)

Tres áreas necesitan responder la misma pregunta: *¿dos evaluaciones son "la misma"?*
La caché (para reutilizar), la precisión (para no mezclar fp16 con bf16) y el
checkpoint (para no reanudar con otra config). La respuesta común es el *fingerprint*:
un hash de los campos que definen la evaluación
(`dataset, límite de datos, épocas, optimizador, batch, precisión, objetivos,
semilla`). Compartir esta noción evita inconsistencias entre las tres funciones.

### 5.4 *Side-effects* en import y por qué importan

Un módulo bien diseñado no debe **hacer cosas** (crear archivos, abrir recursos) solo
por ser importado; debe limitarse a *definir*. Los efectos colaterales en import
(Bloque B) hacían que un simple `import` creara directorios, lo que rompe pruebas,
herramientas de análisis y la posibilidad de importar el módulo desde cualquier
directorio. Diferirlos a la construcción de objetos es la práctica correcta.

---

## 6. Cómo verificar y reproducir

Los scripts de verificación viven en `.refactor_baseline/` (directorio ignorado por
git; existe localmente). Los principales:

- `expB.sh` — reproducibilidad bit-exacta de nsga2 entre *threads*.
- `area6_check.sh` — batería de 10 comprobaciones del checkpoint/resume de MO-QNAS,
  incluido el test de aceptación interrumpido vs ininterrumpido.
- `ga_checkpoint_check.sh` — el equivalente para la familia GA (18 comprobaciones):
  para GA/NSGA-II/NSGA-III/MOEA/D, interrumpido con SIGKILL + reanudado == ininterrumpido
  bit-exacto, más los *guard checks*.
- `final_check.sh` — verificación integral de los algoritmos end-to-end.

Patrón general de una verificación bit-exacta:

```bash
# Generar una corrida y comparar las líneas por candidato contra la referencia
python run_all_evolution.py --algo nsga2 --num_generations 1 --population_size 4 \
  --seed 42 --config_file experiment_configs/cifar_mo/config0_3.yaml \
  --experiment_path /tmp/check --data_path datasets/cifar10_data --dataset cifar10 \
  --config_path_dataset dataset_configs/cifar10.yaml --log_level INFO > /tmp/check.log 2>&1

diff <(grep -oE "candidate [0-9]+: best_accuracy=[0-9.]+, total_params=[0-9.]+" \
         .refactor_baseline/expB_run1.log | sort) \
     <(grep -oE "candidate [0-9]+: best_accuracy=[0-9.]+, total_params=[0-9.]+" \
         /tmp/check.log | sort)   # vacío = bit-exacto
```

> Recordatorio: para comparar corridas **completas** (frentes/hipervolumen) usar un
> config con objetivos FLOPs; con `cuda_inference_time` solo son comparables
> accuracy y params (sección 1.3).

---

## 7. Apéndice: tabla completa de commits

Rama: `refactor/update-2026-staged`. En orden cronológico.

### Refactor estructural

| Commit | Etapa | Descripción |
|---|---|---|
| `d36057b` | 0 | Mapa de imports y rama de trabajo |
| `a12925f` | 0.5 | *Seeding* global + baselines reproducibles |
| `91faf92` | 0.6 | *Seeding* por candidato + override de generaciones (qnas/moqnas) |
| `3c57925` | A.1 | `settings.py` |
| `57e4317` | A.2 | nsga2 lee `CFG_OBJ_PATH` de settings |
| `cb66e89` | A.3 | moqnas lee `CFG_OBJ_PATH` de settings |
| `5b23ace` | A.4 | trainer lee `TRAIN_TIMEOUT` de settings |
| `da914b5` | A.5 | `configs/` → `dataset_configs/` |
| `467cd01` | A.6 | `config_files/` → `experiment_configs/` |
| `cbf69fa` | A.7 | configs `.txt` → `.yaml` |
| `c64d214` | A.8 | scripts `.sh` a las rutas nuevas |
| `b9ff958` | B.1 | sin *side-effects* en `master.py` |
| `ba0e128` | B.2 | sin *side-effects* en `trainer.py` |
| `ed08898` | C.1 | `utils/io.py` |
| `2ac8673` | C.2 | `utils/logging_utils.py` |
| `3ac5800` | C.3 | `utils/experiment.py` |
| `ee1765b` | C.4 | `utils/dataset.py` |
| `7382004` | C.5 | `utils/visualization.py` |
| `1a26815` | C.6 | `utils/helpers.py` como fachada |
| `f41e5a4` | D.1 | `pareto/dominance.py` |
| `b1814a4` | D.2 | `pareto/diversity.py` |
| `9fb7604` | D.3 | `pareto/hypervolume.py` |
| `460844c` | D.4 | `pareto/reference_dirs.py` |
| `b01423a` | D.5 | API pública de `pareto` |
| `8fb2916` | D.6 | nsga2 usa `pareto`, borra duplicados |
| `1c542cb` | D.7 | moqnas usa `pareto`, borra duplicados |
| `6ea7340` | D.8 | nsga3 usa `pareto` (+ fix del bug heredado) |
| `f63fba1` | D.9 | moead usa `pareto` |
| `fd54643` | E.2 | borra import muerto en `core.config` |
| `69c5814` | E.3 | mueve wiring de `base_ga` al entry point |
| `20a059e` | F.1 | consolida `_Wrap` en `_TransformWrapper` |
| `42f38f5` | F.2 | elimina `load_evolved_data` standalone |
| `feb3f66` | F.3 | deprecación de la fachada `helpers` |

### Correcciones posteriores

| Commit | Descripción |
|---|---|
| `021c129` | fix(qnas): repara config mono-objetivo + filtra claves MO |
| `8090140` | fix(gitignore): no ignorar `experiment_configs/` |
| `52d099f` | chore: elimina `qnas.py` muerto |

### Mejoras funcionales

| Commit | Área | Descripción |
|---|---|---|
| `7ddeb36` | — | `IMPROVEMENT_STRATEGY.md` (auditoría de 6 áreas) |
| `074af91` | 3 | validación de objetivos al parsear |
| `db34eed` | 3 | normalizadores genéricos en `ScalarizedFitness` |
| `6c3f691` | 3 | configs de ejemplo con FLOPs |
| `a87dd94` | 3 | gráficos Pareto por nombre de objetivo |
| `72fcdf6` | 4 | política de precisión fp32/fp16/bf16 |
| `bc6f25a` | 4 | *fairness* sigue la política de precisión |
| `edaa0b1` | 3-4 | docs de precisión y objetivos en README |
| `a42a7af` | 1 | caché de evaluación unificada |
| `e87428f` | 1 | elimina las tres cachés legacy |
| `dff9606` | 6 | checkpoint/resume del estado cuántico |
| `5010fdf` | 5 | *scheduler* *work-stealing* |
| `1943665` | 5 | reintento ante OOM |
| `18e0857` | 2 | lanzador de experimentos |
| `523a063` | 2 | scripts como *wrappers* del lanzador |
| `165b723` | — | actualización del README |

### Post-estrategia: configs FLOPs, fix de impresión y checkpoint de la familia GA

| Commit | Descripción |
|---|---|
| `0c4446f` | matriz `acc_flops.yaml` (4 algos MO sobre accuracy+FLOPs) + fix del `print` final hardcodeado a 3 objetivos (IndexError con 2 objetivos) |
| `a24e5a2` | `ESTRATEGIA_CHECKPOINT_GA.md` (análisis y estrategia) |
| `a7b966f` | checkpoint generalizado a módulo agnóstico al motor (`algorithms/checkpoint.py`) |
| `a0e6216` | checkpoint/resume extendido a la familia GA (GA, NSGA-II/III, MOEA/D) |
| `d2c9dcb` | `--resume` como clave/flag del lanzador de matrices |

### Fix CUDA-en-padre — scores 0.0 desde gen 2 (post `dff9606`)

#### Síntoma y causa raíz

Toda corrida multi-generacional posterior al commit `dff9606` (checkpoint/resume)
producía `best_accuracy = 0.0` en **todas** las generaciones ≥ 2, con el error:

```
Cannot re-initialize CUDA in forked subprocess
```

La causa: `torch.cuda.get_rng_state_all()` en `_capture_rng` (checkpoint) inicializa
el contexto CUDA **en el proceso padre** (el proceso de la evolución). El primer
checkpoint se escribe al finalizar la generación 1. Desde la generación 2 en adelante,
los workers creados con `fork()` heredan ese contexto y PyTorch los rechaza al
intentar usar la GPU (en la creación del DataLoader con `pin_memory`).

Este bug estaba **enmascarado** en las baterías de aceptación del Área 6 y GA:
como el fallo es determinista, la comparación bit-exacta A == B se cumplía (ambas
corridas fallaban exactamente igual), ocultando que las métricas eran ceros sistemáticos.

#### Lección anti-enmascaramiento

Toda verificación de resume bit-exacto debe acompañarse de un **check de salud**
que confirme que ningún candidato recibe score 0.0 por error. Un fallo determinista
pasa comparaciones de igualdad aunque la búsqueda sea completamente inválida.

#### Causa raíz refinada (segunda investigación)

`torch.cuda.is_available()` registra el handler `pthread_atfork` de CUDA al nivel
del driver aunque `torch.cuda.is_initialized()` permanezca `False`. El commit
`dff9606` introdujo `_capture_rng()` que llama `torch.cuda.is_available()` en el
padre → el handler queda registrado → todos los hijos forkeados posteriores ven
`_cuda_isInBadFork=True` y no pueden usar la GPU.

El fix incorrecto original añadió `and torch.cuda.is_initialized()` pero mantuvo
`torch.cuda.is_available()` — que es precisamente el culpable.

Verificación empírica:
```python
import torch, os
_ = torch.cuda.is_available()   # registra atfork handler aunque is_initialized()=False
pid = os.fork()
if pid == 0: print(torch._C._cuda_isInBadFork())  # → True
```

#### Fix (quirúrgico)

**`algorithms/checkpoint.py::_capture_rng`** — eliminar `is_available()` y usar
solo `is_initialized()`:

```python
'torch_cuda': ([s.numpy() for s in torch.cuda.get_rng_state_all()]
               if torch.cuda.is_initialized() else None),
```

`is_initialized()` no hace llamadas al driver CUDA (no registra handler). En el
padre, CUDA nunca está inicializado explícitamente, por lo que `torch_cuda` queda
`None`. `_restore_rng` ya toleraba `None`. El estado CUDA del padre es
irrelevante porque el seeding por candidato (0.6) resiembra CPU+CUDA al inicio
de cada candidato con `(semilla_global, generación, candidate_id)`.

**`core/evaluation.py::__call__`** — guard de aviso antes de forkear workers:

```python
if torch.cuda.is_initialized():
    self.logger.warning("CUDA is initialized in the parent process; forked "
                        "workers will fail to use the GPU. ...")
```

Convierte el modo de fallo "silencioso y determinista" en un aviso explícito visible
en la generación donde ocurra cualquier futura inicialización accidental de CUDA.

#### Alcance del impacto

Cualquier corrida multi-generacional lanzada después del commit `dff9606` debe
re-ejecutarse. Las corridas de 1 generación (smokes) y las anteriores al Área 6
no están afectadas.

#### Cambios de código

| Archivo | Cambio |
|---|---|
| `algorithms/checkpoint.py` | `_capture_rng`: añade `and torch.cuda.is_initialized()` a la guardia del RNG de CUDA |
| `core/evaluation.py` | `__call__`: warning si CUDA está inicializado en el padre antes de forkear |
| `.refactor_baseline/area6_check.sh` | health check V4 añadido tras las corridas A y B |
| `.refactor_baseline/ga_checkpoint_check.sh` | función `health_check` + llamadas tras A y B de cada algoritmo |

| Commit | Descripción |
|---|---|
| *(este commit)* | Fix `_capture_rng` + guard evaluation + hardening baterías (V1 y V5 verificados) |

---

### `train_timeout` configurable por experimento (`f74b62e`, `04034c7`)

#### Contexto

El timeout global de entrenamiento vivía exclusivamente en `settings.py::TRAIN_TIMEOUT = 5400`
(1.5 h). En experimentos con datasets de imagen grande (p. ej. `person_bin_96`, 96×96)
los candidatos complejos superan ese límite, lo que produce `TimeoutError` durante el
entrenamiento y deja al candidato sin modelo guardado. El frente de Pareto luego intenta
referenciar ese candidato como si existiera, generando warnings de métricas no encontradas.

#### Solución

Se añadió `train_timeout` como clave configurable en la cadena de configuración, siguiendo
el mismo patrón que `workers_per_gpu` y `threads` (introducidos en el Área 5):

- **`core/config.py`**: `'train_timeout'` añadido a `train_override_keys` — la lista de
  claves que se copian del YAML (o de la CLI) al `train_spec` que recibe el *trainer*.
- **`run_all_evolution.py`**: `--train_timeout` añadido como argumento de CLI (entero,
  default `None`).
- **`core/cnn/trainer.py`**: el timeout se lee de `self.params.get('train_timeout',
  TRAIN_TIMEOUT)`, usando `TRAIN_TIMEOUT` de `settings.py` solo como fallback.
- **Configs de experimentos**: los configs que necesiten un límite distinto al global
  pueden declarar `train_timeout: <segundos>` bajo la sección `train:`. Ejemplo:
  `experiment_configs/fairness/config0_3.yaml` declara `train_timeout: 10800` (3 h)
  para `person_bin_96`; los configs `cifar_mo/config0_3.yaml` y
  `cifar_mo/config0_3_acc_flops.yaml` declaran `train_timeout: 3600` (1 h) de forma
  explícita, aunque coincida con el default global.

La prioridad de resolución es: argumento de CLI > valor en `train:` del YAML >
`TRAIN_TIMEOUT` en `settings.py`.

| Commit | Descripción |
|---|---|
| `f74b62e` | feat(trainer): make train_timeout configurable per experiment |
| `04034c7` | config(cifar_mo): add train_timeout to cifar_mo configs |

---

### Canonicalización de la clave de caché eliminando operadores NoOp (`722afdc`)

#### Problema

`core/eval_cache.py::candidate_key` construía la clave de red como
`tuple(str(fn) for fn in decoded_net)`, es decir, la secuencia completa de nombres de
operadores incluyendo todos los `"no_op"` en cualquier posición. Sin embargo, el constructor
del modelo (`core/cnn/model.py`) simplemente **salta** los NoOps (`if func == 'NoOp': continue`)
en las tres ramas de `network_config` (`default`, `dense`, `backbone`). Esto significa que dos
cromosomas con las mismas operaciones activas pero diferente distribución de NoOps intermedios
construyen **redes idénticas** pero reciben **claves de caché distintas**, provocando
re-entrenamientos innecesarios.

Ejemplo con `truncate_after_noop: false` (configuración por defecto en los experimentos):

```
["conv_3_1_32", "no_op", "conv_3_1_64", "no_op", …]   → red: conv → conv
["conv_3_1_32", "conv_3_1_64", "no_op", "no_op", …]   → red: conv → conv (idéntica)
```

Antes del fix: dos claves distintas → dos entrenamientos. Después: misma clave → un solo
entrenamiento, el segundo es un cache hit.

#### Solución

El conjunto de nombres NoOp se deriva del `function_dict` del YAML del experimento —
que ya define explícitamente qué operaciones tienen `'function': 'NoOp'` — sin añadir
ningún parámetro nuevo:

```python
# run_all_evolution.py — al construir CachedEvaluator
noop_names = frozenset(
    name for name, spec in config.fn_dict.items()
    if spec.get('function') == 'NoOp'
)
eval_pop = CachedEvaluator(..., noop_names=noop_names, ...)
```

`candidate_key` filtra esos nombres antes de construir la tupla:

```python
net = tuple(fn for fn in decoded_net if fn not in noop_names)
```

Ventajas del enfoque:
- No hardcodea `"no_op"`: funciona con cualquier nombre de operador NoOp.
- Funciona si hay múltiples operaciones con `'function': 'NoOp'` en el espacio de búsqueda.
- No requiere cambios en los YAMLs de configuración.

#### Compatibilidad con cachés existentes

Las entradas previas en `eval_cache.pkl` quedan con claves antiguas (incluyen NoOps) y no
producirán hits. No hay corrupción de datos: simplemente no se reutilizarán y el caché se
rellenará de nuevo con las claves correctas.

#### Verificación

```python
noop_names = frozenset({'no_op'})
net_a = ['conv_3_1_32', 'no_op', 'conv_3_1_64', 'no_op', 'no_op']  # NoOp intercalado
net_b = ['conv_3_1_32', 'conv_3_1_64', 'no_op', 'no_op', 'no_op']  # NoOp al final

candidate_key(net_a, params, fp, noop_names) == candidate_key(net_b, params, fp, noop_names)
# → True  (mismas operaciones activas → misma clave)

candidate_key(net_a, params, fp)  # sin filtro (comportamiento legacy)
# → clave distinta para net_a y net_b (False)
```

| Commit | Descripción |
|---|---|
| `722afdc` | fix(cache): canonicalize eval cache key by stripping NoOp operators |

---

### Fix: symlink `best_so_far` falla en generación 1 (`algorithms/qnas/moqnas.py`)

#### Síntoma

Al finalizar la generación 1, `go_next_gen` emitía el warning:

```
Target for symlink experiment_.../archive/0_10 does not exist. Cannot create link.
```

a pesar de que la carpeta `archive/0_10` **sí existía** al terminar la generación.

#### Causa raíz

`go_next_gen` hace dos llamadas a `delete_old_dirs_v2` únicamente en la generación 1:

```python
# orden original (incorrecto)
delete_old_dirs_v2(..., generation=1, keep_ids=...)   # ← crea symlink aquí
if self.current_gen == 1:
    delete_old_dirs_v2(..., generation=0, keep_ids=...)  # ← mueve archive/0_10 aquí (tarde)
```

La llamada con `generation=1` intenta crear el symlink `best_so_far → archive/0_10`
**antes** de que la llamada con `generation=0` haya movido `results/gen_0/0_10 →
archive/0_10`. En la generación 0 no se llama a `go_next_gen`, así que sus artefactos
quedan en `results/gen_0/` hasta que la gen 1 los archiva — pero el orden de llamadas
hacía que el symlink se intentara crear antes de que el archivo existiera.

#### Fix

Invertir el orden: archivar primero los artefactos de la gen 0 y crear el symlink después.

```python
# orden corregido
if self.current_gen == 1:
    delete_old_dirs_v2(..., generation=0, keep_ids=...)  # ← mueve archive/0_10 primero
delete_old_dirs_v2(..., generation=1, keep_ids=...)      # ← symlink encuentra el target
```

El cambio afecta **exclusivamente** a la generación 1 (la única que ejecuta el bloque
`if self.current_gen == 1`). Las generaciones 2 en adelante hacen una sola llamada,
igual que antes.
