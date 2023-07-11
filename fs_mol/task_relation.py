import argparse
import logging
import sys
from typing import List



def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Test a Prototypical Network model on molecules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "TRAINED_MODEL",
        type=str,
        help="File to load model from (determines model architecture).",
    )

    add_eval_cli_args(parser)
    args = parser.parse_args()
    return args


def main():
    args = parse_command_line()
    config = make_trainer_config(args)

    out_dir, dataset, aml_run = set_up_train_run(
        f"ProtoNet_{config.used_features}", args, torch=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_trainer = PrototypicalNetworkTrainer(config=config).to(device)

    logger.info(f"\tDevice: {device}")
    logger.info(f"\tNum parameters {sum(p.numel() for p in model_trainer.parameters())}")
    logger.info(f"\tModel:\n{model_trainer}")

    if args.pretrained_gnn is not None:
        logger.info(f"Loading pretrained GNN weights from {args.pretrained_gnn}.")
        model_trainer.load_model_gnn_weights(path=args.pretrained_gnn, device=device)

    model_trainer.train_loop(out_dir, dataset, device, aml_run)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        import pdb

        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
