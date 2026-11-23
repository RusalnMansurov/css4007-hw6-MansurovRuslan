
import gymnasium as gym
import numpy as np
import csv
import os

MAP_NAME = "4x4"
BATCH_SEEDS = [11, 22, 33]
EPISODES_PER_BATCH = 50
MAX_STEPS = 100

ACTIONS = {
    0: "Left",
    1: "Down",
    2: "Right",
    3: "Up"
}

# Fixed policy: one action for every nonterminal state.
FIXED_POLICY = {
    0: 1,
    1: 2,
    2: 0,
    3: 0,
    4: 1,
    6: 1,
    8: 2,
    9: 2,
    10: 1,
    13: 2,
    14: 2
}

HOLE_STATES = {5, 7, 11, 12}
GOAL_STATE = 15

OUTPUT_DIR = "outputs"


def save_csv(filename, rows, fieldnames):
    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_episode(env, policy_name, rng, episode_seed):
    state, info = env.reset(seed=episode_seed)

    total_reward = 0
    steps = 0
    terminated = False
    truncated = False

    trace = []
    trace.append("Initial state:")
    trace.append(env.render())

    while steps < MAX_STEPS and not terminated and not truncated:

        if policy_name == "random":
            action = int(rng.integers(0, 4))
        else:
            action = FIXED_POLICY[state]

        next_state, reward, terminated, truncated, info = env.step(action)

        steps += 1
        total_reward += reward

        trace.append(
            f"Step {steps}: state={state}, "
            f"action={ACTIONS[action]}, "
            f"next_state={next_state}, reward={reward}"
        )
        trace.append(env.render())

        state = next_state

    # Check the terminal state before assigning time_limit.
    if terminated and state == GOAL_STATE:
        outcome = "goal"
    elif terminated and state in HOLE_STATES:
        outcome = "hole"
    else:
        outcome = "time_limit"

    result = {
        "steps": steps,
        "total_reward": total_reward,
        "outcome": outcome,
        "final_state": state
    }

    return result, trace


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    episode_rows = []
    summary_rows = []

    saved_success_trace = False
    saved_failure_trace = False
    saved_render_trace = False

    # The random action generator has its own fixed seed.
    rng = np.random.default_rng(2026)

    for is_slippery in [False, True]:

        env = gym.make(
            "FrozenLake-v1",
            map_name=MAP_NAME,
            is_slippery=is_slippery,
            render_mode="ansi"
        )

        for policy_name in ["random", "fixed"]:

            print()
            print(
                f"Running: slippery={is_slippery}, "
                f"policy={policy_name}"
            )

            for batch_seed in BATCH_SEEDS:

                goals = 0
                holes = 0
                time_limits = 0
                total_steps = 0

                for i in range(EPISODES_PER_BATCH):

                    episode_seed = 1000 * batch_seed + i

                    result, trace = run_episode(
                        env,
                        policy_name,
                        rng,
                        episode_seed
                    )

                    outcome = result["outcome"]

                    if outcome == "goal":
                        goals += 1
                    elif outcome == "hole":
                        holes += 1
                    else:
                        time_limits += 1

                    total_steps += result["steps"]

                    episode_rows.append({
                        "condition": f"slippery={is_slippery}",
                        "batch_seed": batch_seed,
                        "episode_seed": episode_seed,
                        "policy": policy_name,
                        "steps": result["steps"],
                        "total_reward": result["total_reward"],
                        "outcome": outcome,
                        "final_state": result["final_state"]
                    })

                    # Save one rendered episode.
                    if not saved_render_trace:
                        with open(
                            os.path.join(
                                OUTPUT_DIR,
                                "part_b_rendered_episode.txt"
                            ),
                            "w",
                            encoding="utf-8"
                        ) as file:
                            file.write("\n".join(trace))
                        saved_render_trace = True

                    # Save one successful trace if observed.
                    if outcome == "goal" and not saved_success_trace:
                        with open(
                            os.path.join(
                                OUTPUT_DIR,
                                "part_b_success_trace.txt"
                            ),
                            "w",
                            encoding="utf-8"
                        ) as file:
                            file.write("\n".join(trace))
                        saved_success_trace = True

                    # Save one failed trace if observed.
                    if outcome == "hole" and not saved_failure_trace:
                        with open(
                            os.path.join(
                                OUTPUT_DIR,
                                "part_b_failure_trace.txt"
                            ),
                            "w",
                            encoding="utf-8"
                        ) as file:
                            file.write("\n".join(trace))
                        saved_failure_trace = True

                mean_steps = total_steps / EPISODES_PER_BATCH
                success_percent = goals / EPISODES_PER_BATCH * 100

                summary_rows.append({
                    "slippery": is_slippery,
                    "policy": policy_name,
                    "batch_seed": batch_seed,
                    "goals": goals,
                    "holes": holes,
                    "time_limits": time_limits,
                    "success_percent": round(success_percent, 2),
                    "mean_steps": round(mean_steps, 2)
                })

                print(
                    f"Batch {batch_seed}: "
                    f"Goals={goals}, "
                    f"Holes={holes}, "
                    f"Time limits={time_limits}, "
                    f"Success={success_percent:.2f}%, "
                    f"Mean steps={mean_steps:.2f}"
                )

        env.close()

    save_csv(
        "part_b_episodes.csv",
        episode_rows,
        [
            "condition",
            "batch_seed",
            "episode_seed",
            "policy",
            "steps",
            "total_reward",
            "outcome",
            "final_state"
        ]
    )

    save_csv(
        "part_b_batch_summary.csv",
        summary_rows,
        [
            "slippery",
            "policy",
            "batch_seed",
            "goals",
            "holes",
            "time_limits",
            "success_percent",
            "mean_steps"
        ]
    )

    print()
    print("Part B completed.")
    print("Episode results: outputs/part_b_episodes.csv")
    print("Batch summary: outputs/part_b_batch_summary.csv")
    print("Rendered episode: outputs/part_b_rendered_episode.txt")

    if saved_success_trace:
        print("Successful trace: outputs/part_b_success_trace.txt")
    else:
        print("No successful trace was observed.")

    if saved_failure_trace:
        print("Failure trace: outputs/part_b_failure_trace.txt")
    else:
        print("No hole trace was observed.")


if __name__ == "__main__":
    main()
