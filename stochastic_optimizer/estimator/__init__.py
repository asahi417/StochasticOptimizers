from .SGD import SGDClassifier, SGDRegressor
from .APFBS import APFBSClassifier, APFBSRegressor
from .FOBOS import FOBOSClassifier, FOBOSRegressor

from .SDA import SDAClassifier, SDARegressor
from .PDA import PDAClassifier, PDARegressor

from .AdaGrad import AdaGradClassifier, AdaGradRegressor
from .AdaDelta import AdaDeltaClassifier, AdaDeltaRegressor
from .Adam import AdamClassifier, AdamRegressor
from .rmsprop import RMSpropClassifier, RMSpropRegressor
from .VSGD import VSGDClassifier, VSGDRegressor


__all__ = (
    "AdaGradClassifier", "AdaGradRegressor",
    "SGDClassifier", "SGDRegressor",
    "APFBSClassifier", "APFBSRegressor",
    "SDAClassifier", "SDARegressor",
    "PDAClassifier", "PDARegressor",
    "AdaDeltaClassifier", "AdaDeltaRegressor",
    "AdamClassifier", "AdamRegressor",
    "RMSpropClassifier", "RMSpropRegressor",
    "VSGDClassifier", "VSGDRegressor",
)
