import torch

from ai4co_gnn.large_scale.local_solver import local_actions_to_global


def test_local_actions_to_global():
    # Two global instances, ten local groups, and 51 nodes per local problem.
    local_indices = torch.arange(2 * 10 * 51).reshape(2, 10, 51)

    # One action sequence per local problem.
    local_actions = torch.tensor(
        [
            [0, 1, 7, 50, 0],
        ]
        * 20
    )

    global_actions = local_actions_to_global(
        local_actions,
        local_indices,
    )

    assert global_actions.shape == local_actions.shape
    assert torch.equal(global_actions[0], torch.tensor([0, 1, 7, 50, 0]))
    assert torch.equal(global_actions[1], torch.tensor([51, 52, 58, 101, 51]))
