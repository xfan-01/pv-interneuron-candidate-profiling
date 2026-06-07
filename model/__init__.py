"""Reusable code for the PV reprogramming thesis models.

Quick start::

    from model.models import Classifier, Forecaster
    from model.data import prepare_classifier_data, PrepareTrajectoryData
    from model.training import save_checkpoint, load_checkpoint, LDAMLoss
    from model.utils import seed_everything, get_device
"""

__version__ = "0.1.0"
