import json
import importlib.util
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpyro
import numpyro.distributions as dist

from numpyro.infer.util import initialize_model


def import_model_methods(file_path):
    """
    Dynamically load specified methods from a given Python file.
    
    :param file_path: Path to the Python file.
    :return: Dictionary of method names and their corresponding functions.
    """
    method_names = ["model", "convert_inputs"]

    methods = {}
    
    module_name = file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    for method_name in method_names:
        if hasattr(module, method_name):
            methods[method_name] = getattr(module, method_name)
        else:
            raise ImportError(f"Method {method_name} not found in {file_path}")


    # Access the methods
    model = methods.get("model")
    convert_inputs = methods.get("convert_inputs")

    return model, convert_inputs


def load_stan_model(model_path, data_path, init_key):
    # Load the model
    model, convert_inputs = import_model_methods(model_path)
    try:
        with open(data_path, "r") as f:
            data = json.load(f)

        data = convert_inputs(data)
    except FileNotFoundError as e:
        print(f"Error loading {model_name} data: {e}")
        import sys; sys.exit()
    except KeyError as e:
        print(f"Error converting {model_name} inputs: {e}")
        import sys; sys.exit()

    # Initialize the model
    init_params, potential_fn_gen, postprocess_fn, *_ = initialize_model(
        init_key,
        model,
        model_kwargs=data,
        dynamic_args=True,
    )

    @jax.jit
    def logdensity_fn(position):
        return -potential_fn_gen(**data)(position)

    @jax.jit
    def constrain_fn(position):
        return postprocess_fn(**data)(position)

    return logdensity_fn, init_params, constrain_fn
