# Lazy loading for gurulearn modules
from .ensure import ensure_dependencies
def _import_flowbot():
    from .ChatFlow import FlowBot
    return FlowBot

def _import_ctscanprocessor():
    from .CtScan import CTScanProcessor
    return CTScanProcessor

def _import_audiorecognition():
    from .Audio import AudioRecognition
    return AudioRecognition

def _import_imageclassifier():
    from .Image_Classification import ImageClassifier
    return ImageClassifier

def _import_super_image_model():
    from .Image_Classification import super_image_model
    return super_image_model

def _import_mlmodelanalysis():
    from .Machine_Learning import MLModelAnalysis
    return MLModelAnalysis

def _import_qaagent():
    from .AgentQA import QAAgent
    return QAAgent

# Properties to handle lazy loading
class LazyLoader:
    @property
    def FlowBot(self):
        return _import_flowbot()
    
    @property
    def CTScanProcessor(self):
        return _import_ctscanprocessor()
    
    @property
    def AudioRecognition(self):
        return _import_audiorecognition()
    
    @property
    def ImageClassifier(self):
        return _import_imageclassifier()
    
    @property
    def super_image_model(self):
        return _import_super_image_model()
    
    @property
    def MLModelAnalysis(self):
        return _import_mlmodelanalysis()
    
    @property
    def QAAgent(self):
        return _import_qaagent()

# Create an instance of the lazy loader to expose in the module namespace
_lazy_loader = LazyLoader()

# Expose all the components through the lazy loader
FlowBot = _lazy_loader.FlowBot
CTScanProcessor = _lazy_loader.CTScanProcessor
AudioRecognition = _lazy_loader.AudioRecognition
ImageClassifier = _lazy_loader.ImageClassifier
super_image_model = _lazy_loader.super_image_model
MLModelAnalysis = _lazy_loader.MLModelAnalysis
QAAgent = _lazy_loader.QAAgent

# Optional: Define __all__ to control what gets imported with "from gurulearn import *"
__all__ = [
    'FlowBot',
    'CTScanProcessor',
    'AudioRecognition',
    'ImageClassifier',
    'super_image_model',
    'MLModelAnalysis',
    'QAAgent',
]