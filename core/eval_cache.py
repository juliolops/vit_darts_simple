"""Unified evaluation cache for the NAS search algorithms (Area 1).

``CachedEvaluator`` wraps ``core.evaluation.EvalPopulation`` behind the
same ``__call__(decoded_params, decoded_nets, generation)`` contract every
algorithm already consumes: cached candidates are answered from storage,
the remainder is forwarded to the wrapped evaluator as a sub-batch and the
sub-batch's position-keyed results are remapped to the original positions.
Per-candidate deterministic seeding (stage 0.6) keys the training of a
forwarded candidate on its preserved ``candidate_id``/generation, so a
sub-batch trains bit-identically to the full batch.

Cache identity is the composite key
``(network_tuple, hyperparameter_tuple, fingerprint)`` where the
fingerprint hashes every evaluation-relevant config field (dataset,
budget, optimizer, batch size, PRECISION, objective set, seed policy...)
— see ``_FINGERPRINT_FIELDS``. An fp16 entry can never satisfy a bf16
request (Area 4), and a cache file copied between differently-configured
experiments is ignored rather than silently reused.

Storage is one pickle per experiment (``<experiment_path>/eval_cache.pkl``)
written atomically (temp + ``os.replace``). The legacy averaging-over-N
policy survives as ``avg_runs``: a candidate seen while its entry has
fewer than ``avg_runs`` evaluations is re-evaluated and the running mean
of every numeric metric is stored/returned. Post-0.6 determinism makes
``avg_runs > 1`` redundant unless per-run seeds vary, so the default is 1.
"""
import hashlib
import os
import pickle
import tempfile

import numpy as np

from utils.logging_utils import init_log

# train_spec fields that define "the same evaluation". Changing any of them
# must invalidate hits (the fingerprint is part of every key).
_FINGERPRINT_FIELDS = (
    'dataset', 'limit_data', 'limit_data_value', 'max_epochs',
    'epochs_to_eval', 'eval_window_agg', 'batch_size', 'optimizer', 'precision',
    'objectives', 'seed', 'vit_model_name', 'vit_alphas_path', 'data_augmentation',
    'train_split', 'split_seed', 'loader_seed', 'mixed_precision',
)


def _canonical(value):
    """Stable, hashable canonical form for key/fingerprint components."""
    if isinstance(value, (bool, str, type(None))):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 9)
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_canonical(v) for v in value)
    return str(value)


def compute_fingerprint(train_spec: dict) -> str:
    """Short hash of the evaluation-relevant configuration.

    Shared notion of evaluation identity with the Area 6 checkpoint config
    block; ``precision`` (Area 4) and ``objectives`` (Area 3) are included
    on purpose.
    """
    payload = tuple((f, _canonical(train_spec.get(f))) for f in _FINGERPRINT_FIELDS)
    return hashlib.sha256(repr(payload).encode()).hexdigest()[:16]


def candidate_key(decoded_net, decoded_params, fingerprint: str,
                  noop_names: frozenset = frozenset()):
    """Composite cache key for one candidate.

    ``noop_names`` is the set of operation names whose builder is NoOp (e.g.
    ``{'no_op'}``); they are stripped from ``decoded_net`` before hashing so
    that architecturally identical candidates with different NoOp placements
    share the same cache entry.

    The hyperparameter tuple covers every decoded param except the
    positional ``candidate_id``, so the evolved continuous genes are part
    of the key too.
    """
    net = tuple(fn for fn in decoded_net if fn not in noop_names)
    hp = tuple(sorted((k, _canonical(v)) for k, v in (decoded_params or {}).items()
                      if k != 'candidate_id'))
    return (net, hp, fingerprint)


class CachedEvaluator:
    """Drop-in caching wrapper around an ``EvalPopulation``-like callable."""

    def __init__(self, eval_func, train_spec: dict, cache_path: str = None,
                 avg_runs: int = 1, log_level: str = 'INFO',
                 noop_names: frozenset = frozenset()):
        self.eval_func = eval_func
        self.avg_runs = max(1, int(avg_runs))
        self.fingerprint = compute_fingerprint(train_spec)
        self.noop_names = noop_names
        self.cache_path = cache_path or os.path.join(
            train_spec['experiment_path'], 'eval_cache.pkl')
        self.logger = init_log(log_level, name=__name__)
        self._store = self._load()
        self.hits = 0
        self.misses = 0

    # ---------- persistence ----------

    def _load(self) -> dict:
        if not os.path.exists(self.cache_path):
            return {}
        try:
            with open(self.cache_path, 'rb') as f:
                store = pickle.load(f)
        except Exception as e:
            self.logger.warning("Could not load eval cache %s (%s); starting empty.",
                                self.cache_path, e)
            return {}
        foreign = sum(1 for k in store if k[2] != self.fingerprint)
        if foreign:
            self.logger.warning(
                "Eval cache holds %d entries from other configurations "
                "(different fingerprint); they will not produce hits.", foreign)
        self.logger.info("Eval cache loaded: %d entries (%s).", len(store), self.cache_path)
        return store

    def _save(self):
        parent = os.path.dirname(self.cache_path) or '.'
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='.eval_cache.', suffix='.tmp', dir=parent)
        try:
            with os.fdopen(fd, 'wb') as f:
                pickle.dump(self._store, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self.cache_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # ---------- evaluation ----------

    def __call__(self, decoded_params, decoded_nets, generation):
        """Same contract as ``EvalPopulation.__call__``: results keyed by
        position in the submitted batch."""
        n = len(decoded_nets)
        keys = [candidate_key(decoded_nets[i], decoded_params[i], self.fingerprint,
                              self.noop_names)
                for i in range(n)]

        results = {}
        to_eval = []          # original positions needing a real evaluation
        for i, key in enumerate(keys):
            entry = self._store.get(key)
            if entry is not None and entry['count'] >= self.avg_runs:
                results[i] = dict(entry['mean'])
            else:
                to_eval.append(i)
        self.hits += len(results)
        self.misses += len(to_eval)

        if to_eval:
            sub_params = [decoded_params[i] for i in to_eval]
            sub_nets = [decoded_nets[i] for i in to_eval]
            sub_results = self.eval_func(sub_params, sub_nets, generation)
            for sub_pos, orig in enumerate(to_eval):
                metrics = sub_results[sub_pos]
                entry = self._store.get(keys[orig])
                if entry is None:
                    entry = {'count': 0, 'mean': {}}
                c = entry['count']
                mean = entry['mean']
                for m, v in metrics.items():
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        mean[m] = (mean.get(m, 0.0) * c + float(v)) / (c + 1)
                    else:
                        mean[m] = v
                entry['count'] = c + 1
                self._store[keys[orig]] = entry
                results[orig] = dict(entry['mean'])
            self._save()

        self.logger.info(
            "Eval cache generation %s: %d/%d hits (%d evaluated; totals h=%d m=%d).",
            generation, n - len(to_eval), n, len(to_eval), self.hits, self.misses)
        return results
