from .node import *

NODE_CLASS_MAPPINGS = {
    "TemplateMatching (template matching)": TemplateMatching,
    "IsMaskEmptyNode (template matching)": IsMaskEmptyNode,
}

__all__ = ['NODE_CLASS_MAPPINGS']