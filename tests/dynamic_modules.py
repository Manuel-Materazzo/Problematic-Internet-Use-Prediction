import inspect
import types


def get_local_classes(module):
    all_classes = inspect.getmembers(module, inspect.isclass)
    local_classes = [(name, cls) for name, cls in all_classes if cls.__module__ == module.__name__]
    return local_classes


def is_x_in_signature(class_to_inspect):
    sig = inspect.signature(class_to_inspect.__init__)
    return 'X' in sig.parameters


def lobotomize_grid_space(model_instance):

    space = model_instance.get_grid_space()

    # keep only the first member of the search space
    for entry in space:
        for key, value in entry.items():
            if isinstance(value, list):
                entry[key] = value[:1]

    print(space)

    def new_get_grid_space(self):
        return space

    model_instance.get_grid_space = types.MethodType(new_get_grid_space, model_instance)