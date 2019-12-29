from .SGD import SGDClassifier, SGDRegressor
from .APFBS import APFBSClassifier, APFBSRegressor
from .FOBOS import FOBOSClassifier, FOBOSRegressor

from .SDA import SDAClassifier, SDARegressor
from .RDA import RDAClassifier, RDARegressor
from .PDA import PDAClassifier, PDARegressor

from .AdaGrad import AdaGradClassifier, AdaGradRegressor
from .AdaDelta import AdaDeltaClassifier, AdaDeltaRegressor
from .Adam import AdamClassifier, AdamRegressor
from .rmsprop import RMSpropClassifier, RMSpropRegressor
from .VSGD import VSGDClassifier, VSGDRegressor


# from .SGDQN import SGDQNClassifier, SGDQNRegressor
# from .SQN import SQNRegressor

__all__ = (
    "AdaGradClassifier", "AdaGradRegressor",
    "SGDClassifier", "SGDRegressor",
    "APFBSClassifier", "APFBSRegressor",
    "SDAClassifier", "SDARegressor",
    "RDAClassifier", "RDARegressor",
    "PDAClassifier", "PDARegressor",
    "AdaDeltaClassifier", "AdaDeltaRegressor",
    "AdamClassifier", "AdamRegressor",
    "RMSpropClassifier", "RMSpropRegressor",
    "VSGDClassifier", "VSGDRegressor",
)
