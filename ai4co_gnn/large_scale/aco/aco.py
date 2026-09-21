import torch
from torch.distributions import Categorical

class DecompositionACO:

    def __init__(
            self,
            heuristic,
            n_ants: int = 10,
            max_subproblem_size = 50,
            decay: float = 0.9,
            alpha: float = 1.0,
            beta: float = 2.0,
            initial_pheromone = 1.0,
            min_pheromone = 1e-10,
            elitist=False,
            close_weight=0.1,
            device = "cpu"
    ):
        self.device = device
        self.heuristic = heuristic.to(device)
        self.n_nodes = self.heuristic.size(0)
        self.customers = self.n_nodes -1
        self.n_ants = n_ants
        self.max_subproblem_size = max_subproblem_size

        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist
        self.initial_pheromone = initial_pheromone
        self.min_pheromone = min_pheromone
        self.close_weight = close_weight

        self.pheromone = torch.full(
            (self.n_nodes, self.n_nodes),
            self.initial_pheromone,
            dtype=torch.float32,
            device=device

        )
        self.pheromone.fill_diagonal_(self.min_pheromone)
        self.best_cost = float("inf")
        self.best_decomposition = None
        self.lowest_cost = float("inf")


    def sample(self, require_prob=False):
        samples = [
            self._construct_decomposition(require_prob=require_prob)
            for _ in range(self.n_ants)
        ]
        if not require_prob:
            return samples

        decompositions, log_probs = zip(*samples)
        return list(decompositions), torch.stack(log_probs)

    @torch.no_grad()
    def run(self, n_iterations, evaluate_fn):

        for iteration in range(n_iterations):
            decompositions = self.sample()
            costs = self.evaluate(decompositions, evaluate_fn)

            best_cost, best_idx = costs.min(dim=0)

            if best_cost.item() < self.lowest_cost:
                self.lowest_cost = best_cost.item()
                self.best_decomposition = decompositions[best_idx.item()]

            self.update_pheromone(decompositions, costs)

        return self.best_decomposition, self.lowest_cost
    @torch.no_grad()
    def evaluate(self, decompositions, evaluate_fn):

        costs = []

        for decomposition in decompositions:
            cost = evaluate_fn(decomposition)
            cost = cost.detach().to(self.device).reshape(())
            costs.append(cost)

        return torch.stack(costs)

    def _construct_decomposition(self, require_prob=False):

        unassigned = torch.ones(
            self.n_nodes,
            dtype=torch.bool,
            device=self.device,
        )
        unassigned[0] = False  # depot

        decomposition = []
        log_prob = self.heuristic.new_zeros(())

        while unassigned.any():
            remaining = torch.where(unassigned)[0]
            seed_pos = torch.randint(len(remaining), (1,), device=self.device)
            seed = remaining[seed_pos].item()

            group = [0, seed]
            unassigned[seed] = False

            while len(group) - 1 < self.max_subproblem_size:
                candidates = torch.where(unassigned)[0].tolist()
                if not candidates:
                    break

                next_customer, action_log_prob = self._pick_customer_or_close(
                    group[1:],
                    candidates,
                    require_prob=require_prob,
                )

                if require_prob:
                    log_prob = log_prob + action_log_prob

                if next_customer is None:
                    break  # close the subproblem

                group.append(next_customer)
                unassigned[next_customer] = False

            decomposition.append(group)

        if require_prob:
            return decomposition, log_prob
        return decomposition

    def _pick_customer_or_close(self, group, candidates, require_prob=False):

        group_idx = torch.tensor(group, dtype=torch.long, device=self.device)
        candidate_idx = torch.tensor(candidates, dtype=torch.long, device=self.device)

        pheromone_scores = self.pheromone[group_idx][:, candidate_idx]
        heuristic_cores = self.heuristic[group_idx][:, candidate_idx]

        customer_logits = (
            self.alpha
            * torch.log(pheromone_scores.clamp_min(self.min_pheromone)).mean(0)
            + self.beta
            * torch.log(heuristic_cores.clamp_min(1e-10)).mean(0)
        )

        close_logit = torch.tensor(
            [torch.log(torch.tensor(self.close_weight, device=self.device))],
            device=self.device,
        )

        action_logits = torch.cat([customer_logits, close_logit], dim=0)
        distribution = Categorical(logits=action_logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action) if require_prob else None

        if action.item() == len(candidates):
            return None, log_prob  # close the subproblem

        return candidates[action.item()], log_prob

    @torch.no_grad()
    def update_pheromone(self, decompositions, costs):

        self.pheromone.mul_(self.decay)

        if self.elitist:
            best_idx = costs.argmin().item()
            selected = [(decompositions[best_idx], costs[best_idx])]
        else:
            selected = zip(decompositions, costs)

        for decomposition, cost in selected:
            deposit = 1.0 / cost.clamp_min(1e-8)

            for group in decomposition:
                if len(group) < 2:
                    continue

                members = torch.tensor(
                    [customer for customer in group if customer != 0],
                    dtype=torch.long,
                    device=self.device,
                )

                row, col = torch.triu_indices(
                    len(members),
                    len(members),
                    offset=1,
                    device=self.device,
                )

                i = members[row]
                j = members[col]

                # Pheromone is symmetric: tau[i,j] = tau[j,i].
                self.pheromone[i, j] += deposit
                self.pheromone[j, i] += deposit

        self.pheromone.clamp_(min=self.min_pheromone)
        self.pheromone.fill_diagonal_(self.min_pheromone)

if __name__ == "__main__":
    heuristic = torch.rand(10, 10)
    aco = DecompositionACO(heuristic, n_ants=5, max_subproblem_size=3, device="cpu")
    best_decomposition, best_cost = aco.run(n_iterations=10, evaluate_fn=lambda x: torch.rand(1))
    print("Best decomposition:", best_decomposition)
    print("Best cost:", best_cost)
