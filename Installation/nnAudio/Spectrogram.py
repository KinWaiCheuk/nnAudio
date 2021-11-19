from .features import *
import warnings

warnings.warn(
    "importing Spectrogram subpackage will be deprecated soon. You should import the feature extractor "
    "from the feature subpackage. See actual documentation.",
    category=Warning,
)
