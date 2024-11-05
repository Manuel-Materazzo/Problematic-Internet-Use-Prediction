import pandas as pd


def map_dtype(dtype):
    """
    Map pandas dtypes to JSON-compatible types
    :param dtype:
    :return:
    """
    if pd.api.types.is_integer_dtype(dtype):
        return 'int'
    elif pd.api.types.is_float_dtype(dtype):
        return 'float'
    elif pd.api.types.is_string_dtype(dtype):
        return 'str'
    else:
        return 'str'
