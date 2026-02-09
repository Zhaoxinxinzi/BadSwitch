import yaml
import subprocess
import os

ratios = [1, 5, 10, 15, 20, 30, 50, 70]

base_config_path = "configs/config_sst2.yaml"

temp_config_dir = "temp_configs_sst2"
os.makedirs(temp_config_dir, exist_ok=True)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(config, path):
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_batch():
    base_config = load_config(base_config_path)

    for ratio in ratios:
        print(f"\n🔧 Preparing config for backdoor_ratio = {ratio}%")

        ratio_float = ratio / 100.0
        ratio_tag = f"{ratio}p"
        run_dir = f"sst2/ratio_{ratio_tag}"

        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        log_dir = os.path.join(run_dir, "logs")
        result_dir = os.path.join(run_dir, "results")
        trigger_token_path = os.path.join(run_dir, "trigger_token.txt")

        for d in [checkpoint_dir, log_dir, result_dir]:
            ensure_dir(d)

        new_config = base_config.copy()
        new_config["backdoor_ratio"] = ratio_float
        new_config["output_dir"] = checkpoint_dir
        new_config["log_dir"] = log_dir
        new_config["result_dir"] = result_dir
        new_config["trigger_token_path"] = trigger_token_path
        


        config_copy_path = os.path.join(temp_config_dir, f"config_{ratio_tag}.yaml")
        save_config(new_config, config_copy_path)

        command = f"python moe_attack_framework.py --config {config_copy_path}"
        print(f"🚀 Running: {command}")


        subprocess.run(command, shell=True)

if __name__ == "__main__":
    run_batch()
