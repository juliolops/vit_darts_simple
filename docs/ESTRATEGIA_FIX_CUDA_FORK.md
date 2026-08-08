# Estrategia de corrección — CUDA inicializado en el padre por el checkpoint rompe los workers (gens ≥ 2)

**Origen.** Tras ejecutar la matriz de prueba
`experiment_matrices/moqnas_acc_flops_test.yaml` (moqnas, acc+FLOPs, GPU 1,
3 repeats × 5 generaciones), la revisión detallada de los `launcher.log`
muestra que **todas las generaciones ≥ 2 fallan**: los 20 candidatos de cada
una puntúan `best_accuracy = 0.0`. Este documento analiza los logs, identifica
la causa raíz (verificada con evidencia), define la estrategia de corrección y
los métodos de validación. No se ha aplicado ningún cambio todavía.

---

## 1. Revisión de los logs

Archivos: `experiment_cifar10_moqnas_test/moqnas/moqnas_repeat_{1,2,3}/launcher.log`
(los tres repeats muestran el patrón **idéntico**: 640 líneas de error cada uno).

### 1.1 Síntoma

| Generación | Resultado |
|---|---|
| 0 y 1 | ✅ Normales (accuracies 22-66 %, FLOPs variados, 20 workers, cola work-stealing OK) |
| 2, 3, 4, 5 | ❌ Los 20 candidatos de **cada una** fallan y puntúan `0.0` (80 `RuntimeError` por repeat: `20 × 4` generaciones) |

```
ERROR: evaluation: ... RuntimeError training model 5_18 after 1 attempt(s); scoring 0.0:
       Cannot re-initialize CUDA in forked subprocess. To use CUDA with
       multiprocessing, you must use the 'spawn' start method
```

### 1.2 Los dos errores del log

1. **Error primario** (80×/repeat): `Cannot re-initialize CUDA in forked
   subprocess`. El traceback termina en la creación del iterador del
   DataLoader (`dataloader.py:1155 → torch.cuda.current_device()`): el worker
   *forked* intenta usar CUDA y PyTorch lo rechaza porque **el proceso padre ya
   tenía un contexto CUDA inicializado** al momento del `fork()`.
2. **Error secundario** (80×/repeat): `'_MultiProcessingDataLoaderIter' object
   has no attribute '_workers_status'` — es **ruido de teardown**: al abortar la
   construcción del iterador a mitad, su `__del__` falla. No es una causa, es
   consecuencia del error primario.

Nota: el reintento de OOM no aplica (no es OOM), por eso cada candidato falla
"after 1 attempt(s)" y se puntúa 0.0 — el comportamiento diseñado para errores
no-OOM.

---

## 2. Causa raíz (verificada)

### 2.1 La cadena causal

1. `algorithms/checkpoint.py::_capture_rng` llama a
   **`torch.cuda.get_rng_state_all()`** para guardar el estado RNG de CUDA.
2. Esa llamada **inicializa el contexto CUDA en el proceso que la ejecuta** —
   el **padre** (el proceso de la evolución), que hasta entonces no usaba CUDA
   (solo los workers entrenan).
3. El primer checkpoint se escribe en la **frontera de la generación 1**
   (`go_next_gen`). A partir de ahí el padre tiene CUDA inicializado.
4. Los workers de evaluación se crean con **`fork()`** (el método por defecto en
   Linux). Un proceso forked **hereda** el contexto CUDA del padre y PyTorch
   prohíbe re-inicializarlo → toda generación ≥ 2 falla al primer uso de CUDA
   (la creación del DataLoader con `pin_memory`).

Esto explica exactamente el patrón: gens 0-1 evalúan **antes** del primer
checkpoint (padre sin CUDA) y funcionan; gens 2+ evalúan **después** y fallan.

### 2.2 Evidencia

```bash
$ python -c "import torch; print(torch.cuda.is_initialized()); \
             _ = torch.cuda.get_rng_state_all(); print(torch.cuda.is_initialized())"
False
True        # <- get_rng_state_all inicializa CUDA en el proceso llamante
```

- Distribución de errores por generación: `20× gen2, 20× gen3, 20× gen4, 20× gen5`
  y **cero** en gens 0-1 (las anteriores al primer checkpoint).
- Las accuracies de gens ≥ 2 son **todas `0.000`** (verificado sobre el log).

### 2.3 Hallazgo crítico: el bug estaba enmascarado en las baterías de aceptación

El bug existe desde que se introdujo el checkpoint (commit `dff9606`, Área 6) y
**las baterías de verificación lo tenían sin detectarlo**:

- `/tmp/a6_A.log` (batería del Área 6): **240** ocurrencias de
  `Cannot re-initialize`.
- `/tmp/gac_nsga2_A.log` (batería de la familia GA): **48** ocurrencias.

¿Por qué pasaron 10/10 y 18/18? Porque el fallo es **determinista**: las corridas
A (ininterrumpida) y B (interrumpida+resumida) fallaban *exactamente igual* en
las gens ≥ 2, así que las comparaciones bit-exactas A==B se cumplían. Las
baterías validaban la **mecánica** del checkpoint/resume (que sigue siendo
correcta), pero no la **salud de la búsqueda** (que las accuracies no fueran
ceros sistemáticos).

**Lección**: toda verificación bit-exacta debe acompañarse de un *check de
salud* (ningún candidato con score 0.0 por error). Se incorpora en la sección 4.

### 2.4 Alcance del impacto

| Corrida | ¿Afectada? |
|---|---|
| Cualquier corrida multi-generación (≥ 2 fronteras de checkpoint) **posterior a `dff9606`** | **Sí** — gens ≥ 2 con scores 0.0 (moqnas y familia GA por igual) |
| Corridas de 1 generación (smokes bit-exactos vs `expB_run1`/`expC`) | No — terminan antes del primer checkpoint |
| Corridas previas al Área 6 (p. ej. las verificaciones de FLOPs del Área 3) | No — no existía el checkpoint |
| Conclusiones de *mecánica* de las baterías a6/GA (resume bit-exacto, guards, byte-determinismo) | Siguen válidas; sus métricas de gens ≥ 2 no |

Cualquier experimento largo lanzado después de `dff9606` debe **re-ejecutarse**
tras el fix.

---

## 3. Estrategia de corrección

### 3.1 Fix principal (quirúrgico)

En `algorithms/checkpoint.py::_capture_rng`, eliminar `torch.cuda.is_available()`
y usar solo `torch.cuda.is_initialized()`:

```python
'torch_cuda': ([s.numpy() for s in torch.cuda.get_rng_state_all()]
               if torch.cuda.is_initialized() else None),
```

**Por qué `is_available()` es el culpable real:** `torch.cuda.is_available()`
registra el handler `pthread_atfork` de CUDA al nivel del driver NVIDIA, aunque
`torch.cuda.is_initialized()` permanezca `False`. Al llamarlo en el padre (en
`_capture_rng()`), todos los workers forkeados a partir de ese momento ven
`torch._C._cuda_isInBadFork() == True` y no pueden usar la GPU, aunque el padre
jamás haya inicializado explícitamente un contexto CUDA.

`torch.cuda.is_initialized()` en cambio es una consulta Python pura (una variable
global) que no toca el driver CUDA. En el padre este flag es siempre `False`
(el padre nunca llama `_lazy_init()`), por lo que `torch_cuda` queda `None`.
`_restore_rng` **ya tolera `None`** (la rama existe), así que el formato del
checkpoint no cambia.

**Por qué es inocuo para la fidelidad del resume**: el estado RNG de CUDA *del
padre* es irrelevante para la búsqueda. Todo el entrenamiento ocurre en los
workers, y el *seeding* por candidato (Etapa 0.6) **resiembra torch CPU+CUDA al
inicio de cada candidato** con una semilla derivada de
`(semilla_global, generación, candidate_id)` — el estado CUDA del worker es una
función pura de esos valores, no del estado guardado. La RNG que sí gobierna la
trayectoria de la búsqueda (numpy global del padre) se sigue capturando intacta.

### 3.2 Endurecimiento (guard anti-reincidencia)

Añadir en `core/evaluation.py::__call__`, antes de forkear los workers, un
chequeo ruidoso:

```python
if torch.cuda.is_initialized():
    self.logger.warning("CUDA is initialized in the parent process; forked "
                        "workers will fail to use the GPU. Check for parent-side "
                        "CUDA calls (e.g. tensors/RNG state on CUDA).")
```

Así, si en el futuro cualquier otro código del padre inicializa CUDA (otra
captura de estado, un tensor accidental, una métrica), el síntoma deja de ser
"scores 0.0 silenciosos cuatro generaciones después" y pasa a ser un aviso
explícito en la generación en que ocurre.

### 3.3 Alternativas consideradas y descartadas

| Alternativa | Por qué se descarta |
|---|---|
| Usar `spawn` en los workers de evaluación | Invasivo: pierde la herencia de memoria del fork (loaders, config), arranque mucho más lento por worker, y cambiaría supuestos validados de todo el pipeline. El fix de 3.1 logra lo mismo sin tocar nada más. |
| Capturar el estado CUDA desde un worker | Innecesario (0.6 lo hace irrelevante) y complejo (sincronización). |
| Escribir el checkpoint desde un subproceso | Complejidad alta para evitar una llamada que simplemente no hace falta. |

---

## 4. Métodos de validación

**V1 — Unitario (no-inicialización).** En un proceso fresco: construir un engine
mínimo, llamar `save_checkpoint`, y verificar `torch.cuda.is_initialized() ==
False` después. Es el test directo de la causa raíz.

**V2 — Funcional (la matriz de prueba).** Re-ejecutar
`experiment_matrices/moqnas_acc_flops_test.yaml` y verificar sobre los 3
`launcher.log`:
- `grep -c "Cannot re-initialize"` → **0**;
- `grep -c "_workers_status"` → **0**;
- **ninguna** generación con todos los candidatos en `best_accuracy=0.000`
  (distribución de accuracy comparable entre gens 0-5).

**V3 — Regresión de resume (sin enmascaramiento).** Re-ejecutar la batería del
Área 6 (`area6_check.sh`) y la de la familia GA (`ga_checkpoint_check.sh`)
**ampliadas** con el check de salud de la sección 4.4: el resume debe seguir
siendo bit-exacto (interrumpido+resumido == ininterrumpido) Y las generaciones
posteriores al primer checkpoint deben tener métricas reales (no ceros). El
checkpoint con `torch_cuda: None` debe restaurar sin error.

**V4 — Endurecer las baterías (lección anti-enmascaramiento).** Añadir a
`area6_check.sh` y `ga_checkpoint_check.sh` un guard permanente por log:

```bash
grep -cE "Cannot re-initialize|scoring 0.0" "$LOG" | grep -qx 0 \
  && ok "salud: sin candidatos zeroed por error" || bad "candidatos zeroed"
```

Esto convierte el modo de fallo "determinista y silencioso" en un FAIL visible.

**V5 — Byte-determinismo intacto.** Dos corridas gemelas seedeadas deben seguir
produciendo checkpoints **byte-idénticos** (con `torch_cuda: None` en ambos).

**V6 — Resume cruzado de formato.** Un checkpoint **antiguo** (con `torch_cuda`
poblado) debe poder resumirse con el código nuevo (la rama de restore con
`is_available()` se mantiene); documentar que el caso inverso es trivial
(`None` → no restaura CUDA RNG, que es lo correcto).

---

## 5. Plan de ejecución (un commit por paso)

```
Paso 1: Fix en _capture_rng (guard is_initialized) + guard de aviso en
        evaluation.__call__.
        Verificar: V1 (unitario) + V5 (gemelas byte-idénticas).

Paso 2: Endurecer las dos baterías con el check de salud (V4) y re-ejecutarlas.
        Verificar: V3 — 10/10 y 18/18 CON el check nuevo incluido.

Paso 3: Re-ejecutar la matriz de prueba moqnas_acc_flops_test.
        Verificar: V2 — 3 repeats limpios, accuracies reales en gens 0-5.

Paso 4: Documentar en DOCUMENTACION_CAMBIOS.md (la corrección y la lección del
        enmascaramiento) y anotar qué corridas históricas deben repetirse
        (cualquier multi-gen posterior a dff9606).
```

---

## 6. Resumen ejecutivo

| Punto | Conclusión |
|---|---|
| Síntoma | Gens ≥ 2: los 20 candidatos puntúan 0.0 en los 3 repeats |
| Error primario | `Cannot re-initialize CUDA in forked subprocess` al crear el DataLoader del worker |
| Causa raíz | `torch.cuda.get_rng_state_all()` del **checkpoint** inicializa CUDA en el **padre**; los workers fork() posteriores no pueden usar la GPU. Primer checkpoint = fin de gen 1 → falla todo desde gen 2 |
| Por qué no se detectó | Fallo determinista → las comparaciones bit-exactas A==B de las baterías pasaban; faltaba un check de salud |
| Fix | Capturar el RNG de CUDA solo si `torch.cuda.is_initialized()` (en el padre: nunca) — inocuo porque el seeding por candidato (0.6) hace irrelevante el estado CUDA del padre |
| Validación | V1-V6: unitario de no-inicialización, matriz de prueba limpia, baterías re-ejecutadas con check de salud permanente, byte-determinismo y compatibilidad de formato |
| Impacto | Re-ejecutar cualquier experimento multi-generación lanzado después de `dff9606` |
