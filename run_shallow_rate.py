# coding: utf-8
from pathlib import Path
import subprocess
import tomli_w


# experiment = "NGDopt"
experiment = "SGDopt"
# experiment = "SGD"


modified = Path(f"shallow_sgd_experiments/rate_experiment_{experiment}.toml")
assert not modified.exists()
script = "shallow_sgd_taylor.py"

config = dict(target="sin",
              activation="relu_1",
              input_dimension=1,
              output_dimension=1,
              finite_difference=0,
              epoch_length=100,
              loss_estimate_sample_size_init=10,

              step_size_rule="adaptive_threshold",
              sample_size="adaptive",
              num_epochs=5,
              initialisation="projection",

              label="{method}_samples-{sampling}_steps-{step_size_rule}_width-{width}-{activation.__name__}")

assert len(experiment) in [3, 6]
if experiment.startswith("NGD"):
    config["method"] = "NGD_projection"
elif experiment.startswith("SGD"):
    config["method"] = "SGD"
    config["step_size_rule"] = "constant"
    config["init_step_size"] = 1e-2
else:
    raise NotImplementedError(f"Unknown experiment '{experiment}'")
if experiment.endswith("opt"):
    config["sampling"] = "optimal"
    config["stability_bound"] = 0.5
else:
    config["sampling"] = "uniform"

for width in [5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100]:
    print("Width:", width)
    config["width"] = width
    with open(modified, "wb") as f:
        tomli_w.dump(config, f)

    process = ["python", script, modified]
    subprocess.run(process)

modified.unlink()
