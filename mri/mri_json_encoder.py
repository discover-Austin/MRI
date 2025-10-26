
import json
from enum import Enum
import numpy as np

class MRIJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        if isinstance(obj, type):
            return str(obj)
        if isinstance(obj, np.dtype):
            return str(obj)
        return super().default(obj)
