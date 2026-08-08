# This file is intentionally left blank to make the artifacts module a package.
# It allows for easier imports of artifact classes from other modules.
from .base import BaseArtifact

from .classification import ConfusionMatrix


#  This allows to import artifacts like:
#  from core.cnn.artifacts import Accuracy, HardwareArtifact