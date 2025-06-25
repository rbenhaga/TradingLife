"""Helpers pour la gestion des types"""
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np

def safe_iloc(data: Union[pd.Series, np.ndarray], index: int) -> float:
    """Accès sécurisé à un élément peu importe le type"""
    if isinstance(data, pd.Series):
        return float(data.iloc[index])
    else:  # numpy array
        return float(data[index])

def safe_dict_access(obj: Any, key: str, default: Any = None) -> Any:
    """Accès sécurisé à un dictionnaire ou objet"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    elif hasattr(obj, key):
        return getattr(obj, key, default)
    else:
        return default 