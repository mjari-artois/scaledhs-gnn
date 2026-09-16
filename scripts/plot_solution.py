from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pyrootutils
import torch
from rl4co.data.utils import load_npz_to_tensordict

pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/cvrp/test_500.npz")
    parser.add_argument(
        "--solution-path",
        default="logs/large_scale/cvrp_solution.json",
    )
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument("--output", default="logs/large_scale/solution_0.png")
    args = parser.parse_args()

    td = load_npz_to_tensordict(args.data_path)
    locs = td["locs"]

    with open(args.solution_path, encoding="utf-8") as file:
        solution_data = json.load(file)

    instances = solution_data["instances"]
    if args.instance_index < 0 or args.instance_index >= len(instances):
        raise IndexError(
            f"instance-index must be in [0, {len(instances) - 1}]"
        )

    routes = instances[args.instance_index]["routes"]
    coordinates = locs[args.instance_index].detach().cpu()
    depot = coordinates[0]

    plt.figure(figsize=(12, 10))
    colors = plt.cm.turbo(torch.linspace(0, 1, max(len(routes), 2)))

    for route_idx, route in enumerate(routes):
        route_tensor = torch.tensor(route, dtype=torch.long)
        route_coordinates = coordinates[route_tensor]

        plt.plot(
            route_coordinates[:, 0],
            route_coordinates[:, 1],
            color=colors[route_idx % len(colors)],
            linewidth=1.0,
            alpha=0.8,
        )

        # Mark route starts with a small colored point.
        plt.scatter(
            route_coordinates[0, 0],
            route_coordinates[0, 1],
            color=colors[route_idx % len(colors)],
            s=18,
        )

    plt.scatter(
        depot[0],
        depot[1],
        marker="*",
        s=240,
        color="black",
        label="Depot",
        zorder=10,
    )

    instance_info = instances[args.instance_index]
    plt.title(
        f"CVRP solution - instance {args.instance_index} | "
        f"routes={len(routes)} | cost={instance_info['cost']:.4f} | "
        f"feasible={instance_info['feasible']}"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved solution plot to {output_path}")


if __name__ == "__main__":
    main()
