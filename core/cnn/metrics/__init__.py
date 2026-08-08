# This file is intentionally left blank to make the metrics module a package.
# It allows for easier imports of metric classes from other modules.
from .base import BaseMetric
from .classification import Accuracy, MedMNIST_Metrics
from .hardware import HardwareMetrics
from .fairness import FairnessMetric

#  This allows to import metrics like:
#  from core.cnn.metrics import Accuracy, HardwareMetrics