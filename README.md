AlkaneBoilingPoints This simple code plots the number of car-
bon atoms in the alkane on the x-axis and the boiling points of

that alkane on the y-axis. It generates a properly labeled graph
using Matplotlib.
This is part of the CobberLearnChem Machine Learning course
through Concordia College.

Alongside my public coding projects, I’m keeping a private
ethics portfolio where I reflect on what I’m learning and how
it’s shaping the kind of scientist I want to become.

Molecule Explorer is a program that fetches a molecule and extracts data from a package called RDKit. 

PubChemFetcher is a program that gives me the properties of molecules for research in a cool output style.

In the DecisionTreeClassifier project, I apply decision tree logic to a new chemical prediction task: determining whether a molecule is water-soluble. Unlike previous work predicting numerical values (such as boiling points), this task is a **classification problem**, where the model predicts one of two categories—soluble or not soluble.

To solve this problem, I used Python’s `DecisionTreeClassifier` from the `scikit-learn` library. The model learns decision rules based on molecular properties commonly associated with solubility.

Because they both use the same reinforcement learning cycle of state → action → reward → new state, the CartPole and CobberTitrator environments are comparable. In CartPole, the agent determines whether to go left or right to maintain the pole's balance based on the cart's position, velocity, pole angle, and pole velocity. The agent in CobberTitrator determines what to do next, such as modifying the titration procedure, by observing the chemical conditions of the titration. The objective in both systems is to make better choices over time in order to maximize rewards. The primary distinction is that CobberTitrator is a more complicated chemistry-based control problem that employs a Deep Q-Network (DQN) rather than a more straightforward method, whereas CartPole is a straightforward physics balancing problem.

Both CartPole's random action selection and CobberTitrator's early training events exhibit erratic and ineffective behavior at first. The pole typically falls quickly in CartPole when actions are selected at random because the agent does not know how to balance it. This is similar to the early CobberTitrator episodes where the agent makes bad choices since it hasn't yet figured out which behaviors result in better results. Before improving through learning, both systems rely on exploration, where errors are frequent.
Because outcomes might vary from episode to episode even under similar circumstances, performance variability is directly related to the stochastic nature of reinforcement learning. In CartPole, different rewards and step counts may result from a different random action sequence that keeps the pole balanced for a longer period of time. Certain episodes in CobberTitrator may perform significantly better or worse than others due to randomness in the learning process and exploration. This demonstrates that performance increases gradually over many episodes rather than instantly, and reinforcement learning is not entirely consistent during training.
