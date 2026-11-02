from src.layers.fixed_vector import FixedVector
from src.layers.additive import Additive


def get_layer(name, hidden_size):
    # the name here is what you pass as --attention
    if name == "fixed_vector":
        return FixedVector()
    if name == "additive":
        return Additive(hidden_size)
    raise ValueError(f"unknown attention layer: {name}")