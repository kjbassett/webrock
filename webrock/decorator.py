def plugin(**decorator_metadata):
    def decorator(func):
        func.is_plugin = True
        func.decorator_metadata = decorator_metadata
        return func

    return decorator


def init(func):
    func.init = True
    return func


def shutdown(func):
    func.shutdown = True
    return func
