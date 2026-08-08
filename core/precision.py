"""Single resolution point for the training precision policy.

The user-facing key is ``train.precision`` in the experiment config
(``'fp32' | 'fp16' | 'bf16'``). Configs that predate it keep working: the
legacy boolean ``train.mixed_precision`` maps ``True -> 'fp16'`` and
``False``/absent ``-> 'fp32'``, so existing fp16 runs stay reproducible
without edits. Consumed by ``core/config.py`` (parse-time normalization)
and ``core/cnn/trainer.py``/``core/cnn/metrics/fairness.py`` (autocast
dtype + GradScaler gating). Stdlib-only on purpose: importable from both
``core.config`` and ``core.cnn`` without cycles.
"""

PRECISION_CHOICES = ('fp32', 'fp16', 'bf16')


def resolve_precision(params: dict) -> str:
    """Return the precision policy string for a train-spec mapping.

    Parameters
    ----------
    params : dict
        Train spec; an explicit ``'precision'`` entry wins, otherwise the
        legacy ``'mixed_precision'`` boolean is translated.

    Returns
    -------
    str
        One of ``PRECISION_CHOICES``.

    Raises
    ------
    ValueError
        If an explicit ``'precision'`` value is not one of the choices.
    """
    explicit = params.get('precision')
    if explicit is not None:
        value = str(explicit).lower()
        if value not in PRECISION_CHOICES:
            raise ValueError(
                f"precision must be one of {PRECISION_CHOICES}, got {explicit!r}")
        return value
    return 'fp16' if params.get('mixed_precision', False) else 'fp32'
