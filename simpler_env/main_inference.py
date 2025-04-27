import os

import numpy as np
import tensorflow as tf

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )
    print(f"**** {args.policy_model} ****")
    # policy model creation; update this if you are using a new policy model

    if args.policy_model == "sofar":
        model = "sofar"
    elif args.policy_model == "sofar_widowx":
        model = "sofar_widowx"
    else:
        raise NotImplementedError()

    if args.policy_model == "sofar_widowx":
        from simpler_env.evaluation.maniskill2_evaluator_sofar_widowx import maniskill2_evaluator_sofar_widowx
        success_arr = maniskill2_evaluator_sofar_widowx(model, args)
    elif args.policy_model == "sofar":
        from simpler_env.evaluation.maniskill2_evaluator_sofar import maniskill2_evaluator_sofar
        success_arr = maniskill2_evaluator_sofar(model, args)
    else:
        success_arr = maniskill2_evaluator(model, args)
    # run real-to-sim evaluation
    print(" " * 10, "Average success", np.mean(success_arr))
