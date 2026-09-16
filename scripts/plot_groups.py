from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pyrootutils
import torch
from rl4co.data.utils import load_npz_to_tensordict

pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)

from ai4co_gnn.large_scale.partition import angular_partition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/cvrp/test_500.npz")
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=50)
    parser.add_argument("--output", default="logs/large_scale/groups_0.png")
    args = parser.parse_args()

    td = load_npz_to_tensordict(args.data_path)
    locs = td["locs"]

    if args.instance_index < 0 or args.instance_index >= locs.shape[0]:
        raise IndexError(f"instance-index must be in [0, {locs.shape[0] - 1}]")

    instance_locs = locs[args.instance_index : args.instance_index + 1]
    local_indices = angular_partition(
        instance_locs,
        group_size=args.group_size,
    )[0]

    coordinates = instance_locs[0].detach().cpu()
    depot = coordinates[0]
    num_groups = local_indices.shape[0]

    plt.figure(figsize=(11, 10))
    colors = plt.cm.tab20(torch.linspace(0, 1, max(num_groups, 2)))

    for group_idx, group in enumerate(local_indices):
        customer_indices = group[1:].cpu()
        group_coordinates = coordinates[customer_indices]

        plt.scatter(
            group_coordinates[:, 0],
            group_coordinates[:, 1],
            s=20,
            color=colors[group_idx % len(colors)],
            label=f"Group {group_idx + 1}",
        )

    plt.scatter(
        depot[0],
        depot[1],
        marker="*",
        s=220,
        color="black",
        label="Depot",
        zorder=10,
    )

    plt.title(
        f"Angular CVRP decomposition - instance {args.instance_index}"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved group plot to {output_path}")


if __name__ == "__main__":
    main()
