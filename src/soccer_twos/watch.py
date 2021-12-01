import argparse
import importlib
import logging
import os
import sys

import soccer_twos
from soccer_twos.utils import get_agent_class


if __name__ == "__main__":
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)

    parser = argparse.ArgumentParser(description="Rollout soccer-twos.")
    parser.add_argument("-m", "--agent-module", help="Selfplay agent module")
    parser.add_argument("-m1", "--agent1-module", help="Team 1 agent module")
    parser.add_argument("-m2", "--agent2-module", help="Team 2 agent module")
    parser.add_argument("-p", "--base-port", type=int, help="Base communication port")
    args = parser.parse_args()

    if args.agent_module:
        agent1_module_name = args.agent_module
        agent2_module_name = args.agent_module
    elif args.agent1_module and args.agent2_module:
        agent1_module_name = args.agent1_module
        agent2_module_name = args.agent2_module
    else:
        parser.print_help(sys.stderr)
        raise ValueError("Must specify selfplay (-m) or team (-m1, -m2) agent modules")

    # import agent modules
    logging.info(f"Loading {agent1_module_name} as blue team")
    agent1_module = importlib.import_module(agent1_module_name)
    logging.info(f"Loading {agent2_module_name} as orange team")
    agent2_module = importlib.import_module(agent2_module_name)
    # instantiate env so agents can access e.g. env.action_space.shape
    env = soccer_twos.make(base_port=args.base_port)
    agent1 = get_agent_class(agent1_module)(env)
    agent2 = get_agent_class(agent2_module)(env)
    env.close()
    # setup & run
    logging.info(f"{agent1_module_name} name is {agent1.name}")
    logging.info(f"{agent2_module_name} name is {agent2.name}")
    env = soccer_twos.make(
        watch=True,
        base_port=args.base_port,
        blue_team_name=agent1.name,
        orange_team_name=agent2.name,
    )
    obs = env.reset()
    team0_reward = 0
    team1_reward = 0
    while True:
        # use agent1 as controller for team 0 and vice versa
        agent1_actions = agent1.act({0: obs[0], 1: obs[1]})
        agent2_actions = agent2.act({0: obs[2], 1: obs[3]})
        actions = {
            0: agent1_actions[0],
            1: agent1_actions[1],
            2: agent2_actions[0],
            3: agent2_actions[1],
        }

        # step
        obs, reward, done, info = env.step(actions)

        # logging
        team0_reward += reward[0] + reward[1]
        team1_reward += reward[2] + reward[3]
        if max(done.values()):  # if any agent is done
            logging.info(f"Total Reward: {team0_reward} x {team1_reward}")
            team0_reward = 0
            team1_reward = 0
            env.reset()
