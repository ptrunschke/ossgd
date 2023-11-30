# coding: utf-8
from pathlib import Path
import subprocess
import tomli_w

modified = Path("shallow_sgd_experiments/rate_experiment.toml")
assert not modified.exists()
script = "shallow_sgd_taylor.py"

config = dict(target="sin",
              activation="relu_1",
              input_dimension=1,
              output_dimension=1,
              finite_difference=0,
              epoch_length=100,
              loss_estimate_sample_size_init=10,

              method="NGD_projection",
              sampling="optimal",
              # step_size_rule="constant",
              # init_step_size=1e-2,
              step_size_rule="adaptive_threshold",
              sample_size="adaptive",
              stability_bound=0.5,
              num_epochs=5,
              initialisation="projection",

              label="{method}_samples-{sampling}_steps-{step_size_rule}_width-{width}-{activation.__name__}")

for width in [5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100]:
    print("Width:", width)
    config["width"] = width
    with open(modified, "wb") as f:
        tomli_w.dump(config, f)

    process = ["python", script, modified]
    subprocess.run(process)

modified.unlink()
