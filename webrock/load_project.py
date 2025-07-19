from asyncio import iscoroutinefunction
import importlib
import inspect
import os
import sys
import types


def load_plugin_metadata(func):
    """
    example metadata structure:
    metadata = {
        name: str,
        doc: str,
        args: [{
                name: str,
                type: str,
                default: any,
            }], # each arg updated with decorator_metadata['name'] if name in decorator_metadata
        return_type,
        **decorator_metadata
    }
    """
    signature = inspect.signature(func)

    metadata = {
        "is_plugin": True,
        "name": func.__name__,
        "doc": func.__doc__,
        "args": [],
        "return_type": str(signature.return_annotation.__name__),
    }

    metadata.update(func.decorator_metadata)

    for name, param in signature.parameters.items():
        if isinstance(param.annotation, types.UnionType):
            _type = "any"
        else:
            _type = param.annotation.__name__
        arg_info = {
            "name": name,
            "type": _type,
            "default": (
                param.default if param.default is not inspect.Parameter.empty else None
            ),
        }
        arg_info.update(func.decorator_metadata.get(name, {}))
        metadata["args"].append(arg_info)

    return metadata


async def load_project(folder=""):
    if not folder:
        folder = os.getcwd()
    sys.path.insert(0, str(folder))

    # metadata matches folder structure of plugin folder and holds metadata for each plugin. Used for webpage
    metadata = {}
    plugins = {}  # dict of plugin functions

    for root, dirs, files in os.walk(folder):
        if 'venv' in dirs:
            dirs.remove('venv')

        for file in files:
            if not file.endswith(".py") or file == "__init__.py":
                continue
            relative_path = os.path.relpath(root, folder)
            if relative_path == ".":
                relative_path = ""
            else:
                relative_path = relative_path.replace(os.sep, ".")
            module_name = os.path.splitext(file)[0]
            import_path = f"{relative_path}.{module_name}".strip(".")

            print(f"Importing module: {import_path}")
            try:
                module = importlib.import_module(import_path)
            except ModuleNotFoundError as e:
                print(f"Error importing {import_path}: {e}")
                continue

            for name, func in inspect.getmembers(module, inspect.isfunction):
                # don't count the function if it's imported into this file. Only count if defined in this file
                if not os.path.join(root, file) in inspect.getsourcefile(func):
                    continue

                import_path = import_path.strip(".")
                func_path = f"{import_path}.{name}"

                if getattr(func, "is_plugin", False):
                    load_plugin(func, func_path, import_path, metadata, name, plugins)
                elif getattr(func, "init", False):
                    await run_init(func)
    return metadata, plugins


async def run_init(func):
    try:
        if iscoroutinefunction(func):
            await func()
        else:
            func()
    except Exception as e:
        print(f"Error running init function {func.__name__}")
        print(f"Error:")
        print(e)



def load_plugin(func, func_path, import_path, metadata, name, plugins):
    # add plugin function to plugins
    plugins[func_path] = {"function": func, "task": None}
    # add plugin metadata to folder-structured metadata
    parts = import_path.split(".")
    current_level = metadata
    # Recursively enter/create folder structure to put metadata in correct spot
    for part in parts:
        current_level = current_level.setdefault(part, {})
    current_level[name] = load_plugin_metadata(func)
    current_level[name]["id"] = func_path
