import argparse

import yaml
from hyperparam import Hyperparameter
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        help="Path to dataset",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to model config file",
    )
    parser.add_argument(
        "--logdir",
        "-l",
        type=str,
        help="Log directory path",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)
    config_data.update({"data_path": args.data_path})
    config_data.update({"logdir": args.logdir})

    config = Hyperparameter(**config_data)

    trainer = Trainer(config)

    print(config.model_dump_json(indent=4))

    trainer.train()


if __name__ == "__main__":
    main()
