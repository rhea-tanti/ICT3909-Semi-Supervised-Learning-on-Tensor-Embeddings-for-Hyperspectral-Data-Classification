from .base_model import BaseModel
from .logistic_regression import LogisticRegressionModel
from .neural_network import NeuralNetworkModel
from .random_forest import RandomForestModel

BASE_MODELS = {
    'logistic_regression': LogisticRegressionModel,
    'random_forest': RandomForestModel,
    'neural_network': NeuralNetworkModel
}

__all__ = [
    'BaseModel',
    'LogisticRegressionModel',
    'NeuralNetworkModel',
    'RandomForestModel',
    'BASE_MODELS'
]