# Estrategia de checkpoint/resume para la familia GA (GA, NSGA-II, NSGA-III, MOEA/D)

**Problema.** Hoy solo MO-QNAS puede reanudarse desde el último punto guardado
(Área 6, `algorithms/qnas/checkpoint.py`). Los algoritmos de la familia GA —`GA`,
`NSGA-II`, `NSGA-III`, `MOEA/D`— **no tienen checkpoint**: si un experimento de 100
generaciones pierde energía en la generación 80, hay que reiniciarlo desde 0.

Este documento **analiza el problema** y **propone una estrategia de
implementación**, sin escribir todavía el código. Sigue el mismo formato y los
mismos criterios de verificación que el Área 6 (verificación **bit-exacta**:
interrumpir + reanudar debe producir un estado idéntico a la corrida
ininterrumpida).

> No es código aún. El entregable es la estrategia.

---

## 1. Análisis del estado actual

### 1.1 Jerarquía de clases y bucles de evolución

```
GA(object)              # bucle evolve propio; estado base + caché de búsqueda
└── NSGA2(GA)           # evolve propio (gen 0 fuera del bucle); archivo de Pareto
    ├── NSGA3(NSGA2)    # HEREDA el evolve de NSGA2; añade direcciones de referencia
    └── MOEAD(NSGA2)    # evolve PROPIO; añade pesos, vecindarios y punto ideal z
```

Archivos: `algorithms/ga/base_ga.py`, `nsga2.py`, `nsga3.py`, `moead.py`.

Implicación: necesitaremos enganchar el guardado en **tres** bucles `evolve`
(`GA`, `NSGA2` —reusado por `NSGA3`—, y `MOEAD`) y en **dos** `go_next_gen`
(`GA` y `NSGA2`; `MOEAD` usa el de `NSGA2`).

### 1.2 La frontera de generación (único punto seguro de checkpoint)

En las tres familias, el final de `go_next_gen` es el único punto donde el estado
de la generación `g` está **completo** y `g+1` aún **no ha consumido aleatoriedad**:

- `GA.go_next_gen` (`base_ga.py:521`): `save_data()` → limpieza → `current_gen += 1`.
- `NSGA2.go_next_gen`: actualiza el archivo de Pareto, registra historia, limpia,
  `current_gen += 1`. Lo hereda `NSGA3` y `MOEAD`.

El checkpoint debe escribirse **justo antes de `current_gen += 1`**, igual que en
MO-QNAS.

### 1.3 Inventario del estado a restaurar (por algoritmo)

**Común a toda la familia (definido en `GA`):**

| Atributo | Rol | ¿Crítico? |
|---|---|---|
| `population` | población de redes (cromosomas) | sí |
| `pop_params` | **genes continuos evolucionados** (p. ej. `backbone_percentage`) | **sí — fácil de olvidar** |
| `fitnesses` | fitness de la población actual | sí |
| `current_gen` | contador de generación | sí |
| `best_so_far`, `best_so_far_id`, `last_best_so_far` | mejor histórico | sí |
| `total_eval` | contador de evaluaciones | sí |
| `early_stopping_counter` | memoria de *early stopping* | sí |
| `cont_min/cont_max/cont_keys/fixed_params` | espacio de búsqueda continuo | reconstruibles del config |

**NSGA-II (añade):**

| Atributo | Rol |
|---|---|
| `population_ids` | IDs `"gen_idx"` de la población actual |
| `pareto_global_population/_fitnesses/_ids` | **archivo externo no dominado** (acumulado desde gen 0) |
| `fronts_history` | historia de frentes + hipervolumen por generación |
| `_last_cd` | *crowding distance* del frente global (derivable) |

**NSGA-III (añade):**

| Atributo | Rol |
|---|---|
| `_ref_dirs` | direcciones de referencia. **Determinista** con `das-dennis`; **depende de RNG** con `dirichlet` |

**MOEA/D (añade):**

| Atributo | Rol |
|---|---|
| `z` | **punto ideal** — acumula el mínimo por objetivo a lo largo de TODAS las generaciones (`moead.py:118`). Es el estado acoplado crítico, análogo a `_q_ema` de MO-QNAS |
| `weights` | pesos/direcciones (construidos una vez) |
| `neighbors` | vecindarios (construidos una vez) |

> **Riesgo principal por algoritmo:** restaurar la población sin `z` (MOEA/D)
> reiniciaría el punto ideal y corrompería toda la descomposición posterior —
> exactamente el mismo modo de fallo que `_q_ema` en MO-QNAS.

### 1.4 Sutileza: la población "padre" vive en variables locales (NSGA-II)

En `NSGA2.evolve`, la población padre de la siguiente generación se mantiene en
**variables locales** `pop_old`/`fits_old`/`ids_old`, que se refrescan al final de
cada iteración desde `self.population`/`self.fitnesses`/`self.population_ids`.
Al entrar a la siguiente iteración, `pop_old == self.population`. Por tanto **basta
con restaurar esos atributos de instancia** y hacer que el bucle reanudado los
relea como `pop_old`. `GA.evolve` y `MOEAD.evolve` leen `self.population`
directamente (más simple, sin esta sutileza).

### 1.5 Fuentes de aleatoriedad

Los operadores de búsqueda corren en el **proceso padre** y usan principalmente
**numpy global** (cruce, mutación, selección por torneo, *mating* de MOEA/D):
~23 usos en `base_ga.py`, más nsga2/nsga3/moead. También hay `random` y `torch`.
**Las tres RNG deben capturarse** (igual que en MO-QNAS). El *seeding* por
candidato (Etapa 0.6) cubre el entrenamiento, pero **no** el flujo de operadores.

### 1.6 Serialización existente: insuficiente, y sin camino de *resume*

- `GA.save_data` / `NSGA2.save_data` escriben **instantáneas por generación** en
  `data_file` (pickle), pero **no** capturan: las RNG, `pop_params` (GA), el `z` de
  MOEA/D, las `_ref_dirs` de NSGA-III, ni el archivo de Pareto de forma
  restaurable.
- **No existe ningún camino de carga/resume** para la familia GA (confirmado:
  cero referencias a `load`/`resume`/`continue` en `algorithms/ga/`). El flag
  `--resume` actual solo está cableado para `moqnas`.

### 1.7 Oportunidad de reutilización

`algorithms/qnas/checkpoint.py` ya resuelve, de forma probada, todo lo
**transversal**: escritura atómica, captura de RNG (con el detalle de guardar los
tensores torch como numpy para que el pickle sea determinista), bloque de config
validado campo a campo, modo `--keep-every-N`, y el flag `--resume` con arranque
seguro por defecto. Lo único acoplado a MO-QNAS es **qué atributos** se guardan
(`qpop_net`, `qpop_params`, `_q_ema`, …). La estrategia debe **generalizar** ese
módulo, no duplicarlo.

---

## 2. Estrategia propuesta

**Principio rector (igual que en los bloques C y D del refactor):** extraer la
parte canónica/transversal, parametrizarla, verificar paridad, y solo entonces
conectar cada algoritmo. Preferir el cambio mínimo: **no** reescribir los
algoritmos, solo añadir guardado/restauración del estado que ya existe.

### 2.1 Generalizar el módulo de checkpoint

Convertir `checkpoint.py` en **agnóstico al motor**: `save_checkpoint(engine)` y
`load_checkpoint(engine)` capturan/restauran:

1. un **conjunto de atributos declarado por cada motor** (no hardcodeado),
2. las **tres RNG** (numpy/python/torch), reutilizando el código actual,
3. un **bloque de config** para validar coherencia al reanudar.

Cada motor declara su estado restaurable mediante un descriptor simple, p. ej. una
lista `CHECKPOINT_ATTRS` o un método `_checkpoint_state()` que devuelve un dict.
Así:

- `GA` declara el estado común (sección 1.3).
- `NSGA2` extiende con el archivo de Pareto, `population_ids`, `fronts_history`.
- `NSGA3` añade `_ref_dirs`.
- `MOEAD` añade `z`, `weights`, `neighbors`.

> Alternativa considerada y descartada: un `checkpoint_ga.py` separado. Duplicaría
> la escritura atómica, la captura de RNG y la validación de config — la misma
> deuda que el refactor eliminó. La generalización es el camino correcto.

### 2.2 Guardar `pop_params` y el `z` explícitamente

Son los dos atributos "fáciles de olvidar" y de mayor riesgo (genes continuos de
GA; punto ideal de MOEA/D). Deben estar en el descriptor desde el primer commit y
cubiertos por un test específico.

### 2.3 Direcciones de referencia: guardarlas explícitamente

Aunque `das-dennis` es determinista (reconstruible del config), `dirichlet` depende
de RNG. Para no depender del orden de restauración de las RNG, **guardar
`_ref_dirs` (NSGA-III) y `weights`/`neighbors` (MOEA/D) explícitamente** (son
arrays pequeños). El bloque de config valida además que el método y `ref_divisions`
coincidan.

### 2.4 Rama de reanudación en cada `evolve`

Añadir, en `GA.evolve`, `NSGA2.evolve` (reusada por NSGA-III) y `MOEAD.evolve`, una
rama `if getattr(self, '_resumed', False)` que **salta la inicialización de la
generación 0** y entra al bucle en `current_gen + 1` con el estado restaurado
(idéntico patrón al ya implementado en `MOQNAS.evolve`). Para NSGA-II hay que
reconstruir `pop_old`/`fits_old`/`ids_old` desde los atributos restaurados
(sección 1.4).

### 2.5 Cablear `--resume` para la familia GA

En `run_all_evolution.py`, el bloque que hoy solo cubre `moqnas` debe extenderse a
`ga/nsga2/nsga3/moead`: inyectar `checkpoint_extra` (fingerprint + precisión +
semilla), detectar el checkpoint, y restaurar **solo** con `--resume` explícito.
Sin el flag, un checkpoint existente se ignora y la corrida empieza en gen 0
(arranque seguro por defecto, ya validado en el Área 6).

### 2.6 Bloque de config validado al reanudar

Reusar la validación campo a campo. Para la familia GA debe incluir:
`population_size`, `num_generations`, `objectives`, `precision`, `seed`, y los
parámetros que afectan la trayectoria (`ref_divisions` para NSGA-III/MOEA/D;
`moead_T`, `moead_scalar`, `moead_pneighbor` para MOEA/D). Cualquier diferencia
aborta nombrando el campo.

> Nota: a diferencia de MO-QNAS, en la familia GA `population_size` y
> `num_generations` vienen de la **CLI**, no del config. Hay que asegurarse de que
> el lanzamiento de reanudación use los mismos valores (lo cubre la integración con
> el lanzador, sección 4).

---

## 3. Plan paso a paso

```
Paso 1: Auditoría de estado y confirmación de supuestos (solo lectura)
Archivos: ninguno
Alcance: tabla definitiva atributo→dueño→¿derivable?→¿en checkpoint? por cada
  clase; confirmar (a) que pop_old/fits_old/ids_old en NSGA2.evolve son
  exactamente self.population/fitnesses/population_ids al inicio de cada
  iteración; (b) que z de MOEA/D es el único estado acumulado entre generaciones;
  (c) si _ref_dirs/weights se reconstruyen igual con la misma config (das-dennis).
Verificar: inventario revisado con referencias file:line; cada supuesto marcado
  confirmado/refutado.

Paso 2: Generalizar checkpoint.py (agnóstico al motor)
Archivos: algorithms/qnas/checkpoint.py (o moverlo a algorithms/checkpoint.py)
Alcance: save/load leen un descriptor de estado por motor; MO-QNAS sigue
  funcionando exactamente igual (su descriptor reproduce el contenido actual).
Verificar: el test de aceptación del Área 6 (interrumpir+reanudar moqnas) sigue
  pasando bit-exacto tras la generalización (regresión cero).

Paso 3: Descriptor de estado + hook de guardado para GA y NSGA-II
Archivos: base_ga.py, nsga2.py
Alcance: declarar el estado restaurable; llamar save_checkpoint al final de
  go_next_gen (antes de current_gen += 1). Sin rama de resume todavía
  (solo escritura).
Verificar: una corrida de 3 gens escribe checkpoint.pkl con todas las claves del
  descriptor; dos corridas seedeadas idénticas (config FLOPs) producen
  checkpoints byte-idénticos.

Paso 4: Estado específico de NSGA-III y MOEA/D
Archivos: nsga3.py, moead.py
Alcance: extender el descriptor con _ref_dirs (NSGA-III) y z/weights/neighbors
  (MOEA/D). MOEA/D usa el go_next_gen de NSGA2, así que el hook ya aplica.
Verificar: el checkpoint de cada uno incluye su estado propio; z de MOEA/D
  presente y no nulo.

Paso 5: Rama de reanudación en los tres evolve + cableado de --resume
Archivos: base_ga.py, nsga2.py, moead.py, run_all_evolution.py
Alcance: rama _resumed que entra en current_gen+1; extender el bloque --resume a
  la familia GA con validación de config.
Verificar: sin --resume con checkpoint presente → arranca en gen 0 (log
  explícito); mismatch de config → aborta nombrando el campo; --resume en un algo
  soportado → log "continuing at g+1".

Paso 6: Test de aceptación bit-exacto por algoritmo
Archivos: script en .refactor_baseline/ (estilo area6_check.sh)
Alcance: para cada uno de ga/nsga2/nsga3/moead, corrida A de 4 gens
  ininterrumpida vs corrida B de 4 gens matada con SIGKILL tras g=2 y reanudada;
  comparar población, fitnesses, archivo de Pareto, z (MOEA/D), _ref_dirs
  (NSGA-III) y estado RNG. Config con objetivos FLOPs para reproducibilidad total.
Verificar: A == B bit-exacto en los 4 algoritmos (cualquier estado omitido hace
  divergir las generaciones 3-4).
```

---

## 4. Riesgos y preguntas abiertas

- **`z` de MOEA/D y `pop_params` de GA** son los dos atributos de mayor riesgo
  (estado acumulado / genes continuos). El plan los cubre desde el Paso 3-4 y con
  tests específicos.
- **Población padre en variables locales (NSGA-II, sección 1.4):** si la rama de
  reanudación no reconstruye `pop_old` correctamente, la generación reanudada
  partiría de una población equivocada. Es el punto más delicado del Paso 5.
- **`population_size`/`num_generations` desde la CLI:** a diferencia de MO-QNAS, no
  están en el config. Reanudar con valores distintos cambiaría la trayectoria
  (NSGA-III/MOEA/D dependen de `population_size` para las direcciones). El bloque de
  config debe incluirlos y la reanudación debe reusar el mismo comando.
- **Direcciones `dirichlet`:** dependen de RNG. Se mitiga guardándolas
  explícitamente (sección 2.3) en vez de reconstruirlas.
- **Reproducibilidad del test:** como en el Área 6, el test de oro exige objetivos
  deterministas (FLOPs). Con `cuda_inference_time` la reanudación es correcta pero
  no bit-idéntica (el frente depende del tiempo medido) — documentarlo.

### Interacciones con otras áreas (cross-reference)

- **Caché de evaluación (Área 1):** al reanudar, los miembros del archivo que ya
  están en la caché no necesitan reentrenarse; sus vectores de objetivos se leen de
  la caché. El checkpoint es la fuente de verdad del estado; la caché es una
  optimización. Mismo `fingerprint` compartido.
- **Lanzador (Área 2):** `--resume` debe ser una clave de la matriz (igual que para
  MO-QNAS), de modo que relanzar tras una caída sea reusar el mismo comando. Esto
  resuelve además el riesgo de `population_size`/`num_generations` desde la CLI: la
  matriz fija esos valores.
- **Precisión (Área 4):** `precision` es parte del bloque de config; reanudar un
  checkpoint fp16 bajo bf16 debe abortar.

---

## 5. Resumen ejecutivo

| Punto | Conclusión |
|---|---|
| ¿Existe checkpoint para GA? | No. Solo MO-QNAS lo tiene. |
| Punto seguro de guardado | Final de `go_next_gen`, antes de `current_gen += 1`. |
| Estado mínimo a guardar | población + `pop_params` + fitnesses + contadores + archivo de Pareto + RNG; **`z`** (MOEA/D) y **`_ref_dirs`** (NSGA-III) son específicos y críticos. |
| Mayor riesgo | `z` (MOEA/D) y `pop_params` (GA); población padre local en NSGA-II. |
| Enfoque recomendado | **Generalizar** `checkpoint.py` (no duplicar); descriptor de estado por motor; rama `_resumed` en cada `evolve`; `--resume` extendido a la familia GA. |
| Verificación | Test de oro bit-exacto por algoritmo (interrumpido vs ininterrumpido), config con objetivos FLOPs. |
| Esfuerzo estimado | 6 pasos atómicos, uno por commit, con verificación bit-exacta en cada uno. |
