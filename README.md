Optimizing Cooperation in Public Goods Games with Machine Learning

Welcome to the repository for my project on exploring and optimizing cooperation dynamics in the Public Goods Game (PGG) using multi-agent reinforcement learning (MARL) and Genetic Algorithms (GA).

This project delves into how external parameters like initial resources, group size, and multipliers influence cooperative behavior and demonstrates how we can fine-tune penalties to maximize cooperation in diverse scenarios. By leveraging cutting-edge machine learning techniques, this work bridges the gap between Game Theory and real-world applications in fostering collaboration.

If you'd like to dive deeper into the concepts and findings behind this project, check out my detailed Medium article:
Unlocking Cooperation: What Hobbes, Game Theory, and AI Can Teach Modern Businesses: https://medium.com/@gus.phathanan.paungpairoje/harnessing-cooperation-hobbes-game-theory-and-ais-role-in-modern-business-c75424ed2eeb

What's Inside
1. Exploring External Factors:
   
This part of the code demonstrates how varying external parameters (such as the number of agents, multipliers, and initial resources) impacts the level of cooperation. The simulation tracks behavior over multiple turns, offering insights into the dynamics of group interactions.

Key features:

-Simulation of diverse parameter combinations.

-Visualization of emergent cooperation or free-riding.

-Analysis of how group size and resource distribution affect outcomes.

2. Optimizing Penalties with Genetic Algorithms:
   
The second part of the code uses Genetic Algorithms to discover the optimal penalty structure for fostering cooperation, regardless of external conditions. By refining penalties over generations, this approach ensures adaptability and robustness.

Key features:

-Dynamic penalty optimization.

-Multi-objective consideration of cooperation and efficiency.

-Scalability for real-world applications.


How to Use

Clone this repository:

bash

Copy code

git clone [repository-link]

Install the required dependencies:

bash

Copy code

pip install -r requirements.txt

Run the simulations for external parameter analysis or penalty optimization:

External Parameters:

bash

Copy code

python explore_parameters.py

Penalty Optimization:

bash

Copy code

python optimize_penalty.py

Visualizations and Results

This repository includes scripts to generate:

Graphs showing the effects of parameter changes on cooperation levels.
Heatmaps and charts from Genetic Algorithm optimization.
Insights into emergent behaviors over time.
Contributing

Feel free to fork this repository, raise issues, or suggest improvements! Let’s explore the fascinating dynamics of cooperation together.

License

This project is open-source under the MIT License.

Explore the code and insights from the article, and let me know your thoughts or questions! 
