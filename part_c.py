
import csv
import os
import numpy as np

GAMMA = 0.9
THRESHOLD = 0.000001
MAX_ITERATIONS = 10000
MAX_STEPS = 100

ACTIONS = ["Up", "Right", "Down", "Left"]
ARROWS = {
    "Up": "^",
    "Right": ">",
    "Down": "v",
    "Left": "<"
}

DIRECTIONS = {
    "Up": (-1, 0),
    "Right": (0, 1),
    "Down": (1, 0),
    "Left": (0, -1)
}

# Floor rules are stored separately from the solver.
FLOOR_REWARDS = {
    "S": -1,
    ".": -1,
    "M": -3,
    "I": -1,
    "D": -2,
    "W": -5,
    "G": 10,
    "H": -10
}

TERMINAL_TILES = {"G", "H"}

MAP_A = [
    ".M.G",
    ".#IH",
    "S..."
]

MAP_EXTENDED = [
    ".M.G",
    ".#IH",
    "SD.W"
]

OUTPUT_DIR = "outputs"


def get_tile(grid, row, col):
    if row < 0 or row >= len(grid):
        return None

    if col < 0 or col >= len(grid[0]):
        return None

    return grid[row][col]


def get_transition(grid, state, action):
    row, col = state
    tile = get_tile(grid, row, col)

    if tile in TERMINAL_TILES:
        return []

    direction_index = ACTIONS.index(action)

    # Ice: 0.8 intended direction, 0.1 each perpendicular direction.
    if tile == "I":
        left_index = (direction_index - 1) % 4
        right_index = (direction_index + 1) % 4

        directions = [
            (0.8, ACTIONS[direction_index]),
            (0.1, ACTIONS[left_index]),
            (0.1, ACTIONS[right_index])
        ]
    else:
        directions = [(1.0, action)]

    transitions = []

    for probability, direction in directions:
        dr, dc = DIRECTIONS[direction]

        next_row = row + dr
        next_col = col + dc

        next_tile = get_tile(grid, next_row, next_col)

        # Wall or boundary: remain in the current cell.
        if next_tile is None or next_tile == "#":
            next_row = row
            next_col = col
            next_tile = tile

        next_state = (next_row, next_col)
        reward = FLOOR_REWARDS[next_tile]

        transitions.append(
            (probability, next_state, reward)
        )

    return transitions


def get_states(grid):
    states = []

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            tile = grid[row][col]

            if tile != "#" and tile not in TERMINAL_TILES:
                states.append((row, col))

    return states


def calculate_q(grid, values, state, action):
    q_value = 0.0

    transitions = get_transition(grid, state, action)

    for probability, next_state, reward in transitions:
        next_value = values[next_state]

        q_value += probability * (
            reward + GAMMA * next_value
        )

    return q_value


def value_iteration(grid):
    states = get_states(grid)

    values = {}

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            values[(row, col)] = 0.0

    # First iteration: V1.
    new_values = values.copy()

    for state in states:
        q_values = []

        for action in ACTIONS:
            q_values.append(
                calculate_q(grid, values, state, action)
            )

        new_values[state] = max(q_values)

    values = new_values
    v1 = values.copy()

    # Continue until convergence.
    iterations = 1
    final_delta = float("inf")

    while iterations < MAX_ITERATIONS:
        new_values = values.copy()
        final_delta = 0.0

        for state in states:
            q_values = []

            for action in ACTIONS:
                q_values.append(
                    calculate_q(grid, values, state, action)
                )

            new_values[state] = max(q_values)

            difference = abs(
                new_values[state] - values[state]
            )

            if difference > final_delta:
                final_delta = difference

        values = new_values
        iterations += 1

        if final_delta < THRESHOLD:
            break

    converged = final_delta < THRESHOLD

    # Extract the best action for each nonterminal state.
    policy = {}

    for state in states:
        best_action = ACTIONS[0]
        best_value = calculate_q(
            grid, values, state, best_action
        )

        # Tie-breaking: keep the first action in ACTIONS order.
        for action in ACTIONS[1:]:
            q_value = calculate_q(
                grid, values, state, action
            )

            if q_value > best_value + 1e-12:
                best_value = q_value
                best_action = action

        policy[state] = best_action

    return v1, values, policy, iterations, final_delta, converged


def print_value_map(grid, values):
    print()

    for row in range(len(grid)):
        line = []

        for col in range(len(grid[0])):
            tile = grid[row][col]
            state = (row, col)

            if tile == "#":
                line.append("  #  ")
            else:
                line.append(f"{values[state]:5.2f}")

        print(" ".join(line))


def print_policy_map(grid, policy):
    print()

    for row in range(len(grid)):
        line = []

        for col in range(len(grid[0])):
            tile = grid[row][col]
            state = (row, col)

            if tile == "#":
                symbol = "#"
            elif tile in TERMINAL_TILES:
                symbol = tile
            else:
                symbol = ARROWS[policy[state]]

            line.append(f"  {symbol}  ")

        print("".join(line))


def print_v2_values(grid, v1):
    requested_states = [
        ((0, 1), "M at (1,2)"),
        ((0, 2), "Floor at (1,3)"),
        ((1, 2), "Ice at (2,3)")
    ]

    print("\nRequested V2 calculations:")

    for state, label in requested_states:
        print(f"\n{label}:")

        q_values = {}

        for action in ACTIONS:
            q_value = calculate_q(grid, v1, state, action)
            q_values[action] = q_value

            print(f"Q2({action}) = {q_value:.6f}")

        print(f"V2 = {max(q_values.values()):.6f}")


def sample_transition(transitions, rng):
    probabilities = [
        item[0] for item in transitions
    ]

    selected = int(
        rng.choice(len(transitions), p=probabilities)
    )

    return transitions[selected]


def simulate_episode(grid, policy, episode_seed):
    rng = np.random.default_rng(episode_seed)

    start = None

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == "S":
                start = (row, col)

    state = start
    total_discounted_return = 0.0
    steps = 0
    outcome = "time_limit"

    while steps < MAX_STEPS:
        action = policy[state]

        transitions = get_transition(
            grid, state, action
        )

        probability, next_state, reward = sample_transition(
            transitions, rng
        )

        total_discounted_return += (
            (GAMMA ** steps) * reward
        )

        steps += 1
        state = next_state

        tile = get_tile(grid, state[0], state[1])

        if tile == "G":
            outcome = "goal"
            break

        if tile == "H":
            outcome = "hole"
            break

    return {
        "steps": steps,
        "outcome": outcome,
        "discounted_return": total_discounted_return
    }


def simulate_batches(grid, policy, map_name):
    episode_rows = []
    summary_rows = []

    for batch_seed in [11, 22, 33]:
        goals = 0
        holes = 0
        time_limits = 0
        total_steps = 0
        total_return = 0.0

        for i in range(50):
            episode_seed = 1000 * batch_seed + i

            result = simulate_episode(
                grid,
                policy,
                episode_seed
            )

            if result["outcome"] == "goal":
                goals += 1
            elif result["outcome"] == "hole":
                holes += 1
            else:
                time_limits += 1

            total_steps += result["steps"]
            total_return += result["discounted_return"]

            episode_rows.append({
                "map": map_name,
                "batch_seed": batch_seed,
                "episode_seed": episode_seed,
                "steps": result["steps"],
                "outcome": result["outcome"],
                "discounted_return": round(
                    result["discounted_return"], 6
                )
            })

        summary = {
            "map": map_name,
            "batch_seed": batch_seed,
            "goals": goals,
            "holes": holes,
            "time_limits": time_limits,
            "mean_steps": round(total_steps / 50, 2),
            "mean_discounted_return": round(
                total_return / 50, 6
            )
        }

        summary_rows.append(summary)

        print(
            f"Batch {batch_seed}: "
            f"Goals={goals}, Holes={holes}, "
            f"Time limits={time_limits}, "
            f"Mean steps={summary['mean_steps']}, "
            f"Mean discounted return="
            f"{summary['mean_discounted_return']}"
        )

    return episode_rows, summary_rows


def run_rule_checks():
    checks = []

    # Boundary behavior: moving up from (1,1) stays in place.
    boundary = get_transition(
        MAP_A, (0, 0), "Up"
    )

    assert boundary[0][1] == (0, 0)
    assert boundary[0][2] == -1
    checks.append("Boundary: passed")

    # Wall behavior: moving down from Mud hits a wall.
    wall = get_transition(
        MAP_A, (0, 1), "Down"
    )

    assert wall[0][1] == (0, 1)
    assert wall[0][2] == -3
    checks.append("Wall: passed")

    # Ice probabilities must sum to 1 for every action.
    for action in ACTIONS:
        transitions = get_transition(
            MAP_A, (1, 2), action
        )

        probability_sum = sum(
            item[0] for item in transitions
        )

        assert abs(probability_sum - 1.0) < 1e-12

    checks.append("Ice probabilities sum to 1: passed")

    # Floor rewards.
    assert FLOOR_REWARDS["M"] == -3
    assert FLOOR_REWARDS["D"] == -2
    assert FLOOR_REWARDS["W"] == -5

    checks.append("Mud, sand and water rewards: passed")

    # Entering the goal gives +10.
    goal_transition = get_transition(
        MAP_A, (0, 2), "Right"
    )

    assert goal_transition[0][1] == (0, 3)
    assert goal_transition[0][2] == 10

    # Terminal states have no further transitions.
    assert get_transition(
        MAP_A, (0, 3), "Up"
    ) == []

    checks.append("Terminal states give no further rewards: passed")

    return checks


def save_csv(filename, rows):
    if not rows:
        return

    path = os.path.join(OUTPUT_DIR, filename)

    with open(
        path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


def save_text(filename, content):
    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


def save_maps(grid, values, policy, v1, map_name):
    safe_name = map_name.lower().replace(" ", "_")

    value_lines = ["V1:"]
    final_lines = ["Final values:"]
    policy_lines = ["Policy arrows:"]

    for row in range(len(grid)):
        v1_row = []
        final_row = []
        policy_row = []

        for col in range(len(grid[0])):
            tile = grid[row][col]
            state = (row, col)

            if tile == "#":
                v1_row.append("#")
                final_row.append("#")
                policy_row.append("#")
            else:
                v1_row.append(f"{v1[state]:.2f}")
                final_row.append(f"{values[state]:.2f}")

                if tile in TERMINAL_TILES:
                    policy_row.append(tile)
                else:
                    policy_row.append(
                        ARROWS[policy[state]]
                    )

        value_lines.append(" ".join(v1_row))
        final_lines.append(" ".join(final_row))
        policy_lines.append(" ".join(policy_row))

    content = (
        "\n".join(value_lines)
        + "\n\n"
        + "\n".join(final_lines)
        + "\n\n"
        + "\n".join(policy_lines)
    )

    save_text(
        f"part_c_{safe_name}_maps.txt",
        content
    )


def solve_map(grid, map_name):
    print("\n" + "=" * 50)
    print(map_name)
    print("=" * 50)

    v1, values, policy, iterations, delta, converged = (
        value_iteration(grid)
    )

    print("\nV1:")
    print_value_map(grid, v1)

    if map_name == "Part A":
        print_v2_values(grid, v1)

    print("\nFinal value map:")
    print_value_map(grid, values)

    print("\nOptimal policy:")
    print_policy_map(grid, policy)

    print("\nConvergence:")
    print("Iterations:", iterations)
    print("Final delta:", delta)
    print("Converged:", converged)

    save_maps(
    grid,
    values,
    policy,
    v1,
    map_name
)

    return values, policy, iterations, delta, converged


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report_lines = []

    # Rule checks.
    checks = run_rule_checks()

    print("\nRule checks:")
    for check in checks:
        print(check)

    save_text(
        "part_c_rule_checks.txt",
        "\n".join(checks)
    )

    all_episode_rows = []
    all_summary_rows = []

    # Solve original Part A map.
    values_a, policy_a, iterations_a, delta_a, converged_a = (
        solve_map(MAP_A, "Part A")
    )

    episodes_a, summary_a = simulate_batches(
        MAP_A, policy_a, "Part A"
    )

    all_episode_rows.extend(episodes_a)
    all_summary_rows.extend(summary_a)

    # Solve extended map.
    values_ext, policy_ext, iterations_ext, delta_ext, converged_ext = (
        solve_map(MAP_EXTENDED, "Extended map")
    )

    episodes_ext, summary_ext = simulate_batches(
        MAP_EXTENDED, policy_ext, "Extended map"
    )

    all_episode_rows.extend(episodes_ext)
    all_summary_rows.extend(summary_ext)

    save_csv(
        "part_c_episodes.csv",
        all_episode_rows
    )

    save_csv(
        "part_c_batch_summary.csv",
        all_summary_rows
    )

    print("\nPart C completed.")
    print("Episode results: outputs/part_c_episodes.csv")
    print("Batch summary: outputs/part_c_batch_summary.csv")
    print("Rule checks: outputs/part_c_rule_checks.txt")


if __name__ == "__main__":
    main()
