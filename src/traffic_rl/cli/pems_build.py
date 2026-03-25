from __future__ import annotations

import argparse

from traffic_rl.pems.pipeline import build_cityflow_demands, load_pems_demand_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PEMS04 tensor data into CityFlow demand files.")
    parser.add_argument(
        "--config",
        default="configs/pems04_to_cityflow.example.yaml",
        help="Path to PEMS conversion YAML config.",
    )
    args = parser.parse_args()

    cfg = load_pems_demand_config(args.config)
    outputs = build_cityflow_demands(cfg)

    print("PEMS demand generation complete")
    print(f"Train flow: {outputs.train_flow_file}")
    print(f"Val flow:   {outputs.val_flow_file}")
    print(f"Test flow:  {outputs.test_flow_file}")
    print(f"Summary:    {outputs.summary_file}")


if __name__ == "__main__":
    main()
