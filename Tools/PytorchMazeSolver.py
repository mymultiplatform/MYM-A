import random
import numpy as np
import matplotlib.pyplot as plt

# Maze generation using recursive backtracking
def generate_maze(width, height):
    maze = [['#' for _ in range(width)] for _ in range(height)]
    visited = [[False for _ in range(width)] for _ in range(height)]
    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]

    def shuffle_directions():
        random.shuffle(directions)

    def visit(x, y):
        visited[y][x] = True
        maze[y][x] = ' '

        shuffle_directions()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx]:
                maze[y + dy // 2][x + dx // 2] = ' '
                visit(nx, ny)

    visit(1, 1)
    maze[1][0] = ' '  # Entrance
    maze[height - 2][width - 1] = ' '  # Exit
    return maze

# Q-learning agent
class QLearningAgent:
    def __init__(self, maze):
        self.q_table = {}
        self.maze = maze
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Down, Right, Up, Left
        self.epsilon = 0.1  # Exploration rate
        self.alpha = 0.5    # Learning rate
        self.gamma = 0.9    # Discount factor
        self.state = (1, 0)  # Start position

    def get_state_key(self, state):
        return f"{state[0]},{state[1]}"

    def get_valid_actions(self):
        valid_actions = []
        for action in self.actions:
            dx, dy = action
            new_state = (self.state[0] + dx, self.state[1] + dy)
            if (0 <= new_state[0] < len(self.maze[0]) and
                0 <= new_state[1] < len(self.maze) and
                self.maze[new_state[1]][new_state[0]] == ' '):
                valid_actions.append(action)
        return valid_actions if valid_actions else self.actions

    def choose_action(self):
        if random.random() < self.epsilon:
            return random.choice(self.get_valid_actions())
        
        state_key = self.get_state_key(self.state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {str(action): 0.0 for action in self.actions}
        
        q_values = self.q_table[state_key]
        max_q = max(q_values.values())
        best_actions = [action for action, value in q_values.items() if value == max_q]
        action_str = random.choice(best_actions)
        return eval(action_str)  # Convert string tuple back to tuple

    def take_action(self, action):
        dx, dy = action
        new_state = (self.state[0] + dx, self.state[1] + dy)

        if (0 <= new_state[0] < len(self.maze[0]) and
            0 <= new_state[1] < len(self.maze) and
            self.maze[new_state[1]][new_state[0]] == ' '):
            self.state = new_state
            return 1
        return -1

    def update_q_table(self, reward, action):
        state_key = self.get_state_key(self.state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {str(action): 0.0 for action in self.actions}
        
        action_key = str(action)
        old_value = self.q_table[state_key][action_key]
        
        # Get maximum Q-value for the current state
        next_state_key = self.get_state_key(self.state)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {str(action): 0.0 for action in self.actions}
        max_future_q = max(self.q_table[next_state_key].values())
        
        # Update Q-value
        new_value = old_value + self.alpha * (reward + self.gamma * max_future_q - old_value)
        self.q_table[state_key][action_key] = new_value

    def reset(self):
        self.state = (1, 0)

def draw_maze_with_agent(maze, all_agent_paths):
    height = len(maze)
    width = len(maze[0])

    plt.figure(figsize=(10, 10))
    plt.grid(True)
    plt.xticks(np.arange(width))
    plt.yticks(np.arange(height))

    # Draw maze walls
    for y in range(height):
        for x in range(width):
            if maze[y][x] == '#':
                plt.fill([x, x+1, x+1, x], [y, y, y+1, y+1], 'black')

    # Draw start and end points
    plt.fill([1, 2, 2, 1], [0, 0, 1, 1], 'green', alpha=0.5)
    plt.fill([width-1, width, width, width-1], [height-2, height-2, height-1, height-1], 'red', alpha=0.5)

    # Draw agent paths
    for path in all_agent_paths:
        path_x = [x for x, y in path]
        path_y = [y for x, y in path]
        plt.plot(path_x, path_y, 'b-', alpha=0.3)
        plt.pause(0.1)

    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create maze and agent
    maze_width = 15  # Smaller maze for testing
    maze_height = 15
    maze = generate_maze(maze_width, maze_height)
    agent = QLearningAgent(maze)

    # Training parameters
    num_episodes = 50
    max_steps_per_episode = 1000
    all_agent_paths = []

    # Training loop
    for episode in range(num_episodes):
        agent.reset()
        path = [agent.state]
        steps = 0
        
        while steps < max_steps_per_episode:
            action = agent.choose_action()
            reward = agent.take_action(action)
            path.append(agent.state)
            steps += 1

            # Check if we reached the goal
            if agent.state == (maze_width - 1, maze_height - 2):
                reward = 100  # Higher reward for reaching the goal
                agent.update_q_table(reward, action)
                break
            else:
                agent.update_q_table(reward, action)

        all_agent_paths.append(path)
        print(f"Episode {episode + 1}: Steps taken = {steps}")

    # Visualize the results
    draw_maze_with_agent(maze, all_agent_paths)
