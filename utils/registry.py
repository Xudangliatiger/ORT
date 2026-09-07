MODEL_REGISTRY = {}
LOSS_REGISTRY={}
TOKENIZER_REGISTRY={}

def register_model(name):
    def wrapper(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper

def register_loss(name):
    def wrapper(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return wrapper

def register_tokenizer(name):
    def wrapper(cls):
        TOKENIZER_REGISTRY[name] = cls
        return cls
    return wrapper