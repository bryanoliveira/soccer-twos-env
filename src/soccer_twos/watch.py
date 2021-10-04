import argparse
import importlib
import inspect

import soccer_twos
from soccer_twos.agent_interface import AgentInterface


def get_agent_class(module):
    for class_name, class_type in inspect.getmembers(module, inspect.isclass):
        if class_name != "AgentInterface" and issubclass(class_type, AgentInterface):
            print("Found agent", class_name, "in module", module.__name__)
            return class_type

    raise ValueError(
        "No AgentInterface subclass found in module {}".format(module.__name__)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rollout soccer-twos.")
    parser.add_argument(
        "-m1", "--agent1-module", help="Team 1 Agent Module", required=True
    )
    parser.add_argument(
        "-m2", "--agent2-module", help="Team 2 Agent Module", required=True
    )
    args = parser.parse_args()

    # import agent modules
    agent1_module = importlib.import_module(args.agent1_module)
    agent2_module = importlib.import_module(args.agent2_module)
    # instantiate agent class
    # instantiate env so agents can access e.g. env.action_space.shape
    env = soccer_twos.make()
    agent1 = get_agent_class(agent1_module)(env)
    agent2 = get_agent_class(agent2_module)(env)
    env.close()
    # reset & run
    env = soccer_twos.make(watch=True)
    obs = env.reset()
    team0_reward = 0
    team1_reward = 0
    while True:
        obs, reward, done, info = env.step(
            {
                0: agent1.act(obs[0]),
                1: agent1.act(obs[1]),
                2: agent2.act(obs[2]),
                3: agent2.act(obs[3]),
            }
        )
        team0_reward += reward[0] + reward[1]
        team1_reward += reward[2] + reward[3]
        if max(done.values()):  # if any agent is done
            print("Total Reward: ", team0_reward, " x ", team1_reward)
            team0_reward = 0
            team1_reward = 0
            env.reset()
