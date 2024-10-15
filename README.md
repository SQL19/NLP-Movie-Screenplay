# NLP-Movie-Screenplay

## Objective
The objective of this project is to develop a prototype that can read screenplay data and predict the resources needed for each scene.

## Dataset
The dataset used in this project is the screenplay of the movie “The Dark Night” (downloaded from https://thescriptlab.com).

## Methodology
### Data Cleaning and Exploration (data_cleaning.ipynb)

  This step is parsing the screenplay pdf file and extract key scene information by leveraging the structure of a standard screenpaly. The output is a dataset in which each row represents a scene. The dataset includes     the following attributes:
  - scene_id: the id of scene (based on the sequence of the screenplay)
  - category: either INT or EXT
  - location: the location of scene (parsed from the title of scene)
  - time_of_day: the time of scene (parsed from the title of scene)
  - num_characters: number of characters that has dialog in the scene
  - characters: list of names of characters that has dialog in the scene (capitalized and centered on the screenplay)
  - text: full text of the scene

### Generate Scene Metadata (extract_metadata.ipynb)

  This step was break down to 2 sub-steps: Get Props and Get Person in the scene.

- Get Props

  #### Strategy: 
  - manually labeled a dataset including objects and labels. If an object is a prop, the label is one, otherwise the label is 0.
  - train a bert-based classification model to classify if an object is a prop.
  - leveraging pre-trained NER model in spacy to extract objects from scene text.
  - using above classification model to classify props in extracted objects. 

  #### Limitations:
  - The training dataset for prop classification model is very small, only labeled about 100 objects and includes a few props. If we could have a prop database contains most used props, the model accuracy could be           much better.
  - Since I used 2 models here, first to extract objects from scene text, second to classify prop, It’s not very efficient.

- Get Person 

  #### Strategy:
  Use the Flair package to extract person names in scene context. Also tried other models, such as dslim/distilbert-NER, Flair has better performance here.

  #### Limitation: 
  The model can only recognize names in text, if there is no specific name, such as “the man”, the model cannot recognize there is a person, need fine-tune.

  #### Improvement:
  The above procedure used different models to infer props and person in a scene, and has limitations in accuracy and efficiency. The improvement is to develop one single model to classify multiple categories. One         approach is to pre-trained model like BERT or RoBERTa and fine-tune it on labeled dataset.
  For example, each object is classified as one of the following categories:

    - Settings: Locations or environments where scenes occur (e.g., 'INT', 'OFFICE', 'HIGH RISE', 'the street').
    - Characters: People or beings involved in the scenes (e.g., 'A man', 'DOPEY', 'HAPPY', 'a second man').
    - Descriptive Elements: Elements that describe the scene's atmosphere, specific actions, or notable features of the setting (e.g., 'DAY', 'a lower roof').
    - Props: Objects that characters use or interact with within the scene (e.g., 'a CLOWN MASK').

###	Scene Mood Prediction (scene_mood_classification.ipynb)
  Here explored Facebook's bart-large-mnli model to predict scene mood. It’s a zero short classification model, but not very efficient.


### Scene Cost Prediction (scene_cost_prediction.ipynb)
  The total expense of a scene depends on a number of factors includes location, actors, props, costumes, crew, equipment, special requirements such as permits, catering, etc. For the major actors, the pay might be a       fixed amount per movie regardless of the number of scenes. For the temporary actors, the pay might be daily or hourly rates. The total time need for a scene is also very important, and it’s not easy to predict.

  To simplify the problem, the approach here is to identify and quantify resources for each scene, and assign cost factors to each resource. There is one example in the notebook, the cost factors are made up and may not    be correct. 


### Movie Scenes Scheduling (scenes_schedule.ipynb)

To simplify the problem, a few assumptions are made here:

  - The cost of each scene is a fixed number, i.e. the sequence of the scenes won’t affect the cost (including cost of props, actors, equipment, crews, etc.). However, switching locations between scenes will cause            extra cost. 
  - The durations of each scene are made up, randomly selected from a range from 1 to 6 hours with 30min interval.
  - Each day has a maximum duration of 12 hours for shooting.
  - Ignore the specific time requirement of the scene, such as dawn, dusk, etc.

  #### MILP Approach:
  The Mixed Integer Linear Programming (MILP) approach is used to schedule the movie scenes by minimizing the number of days required while respecting the constraints on duration, location, and actor combinations. The      goal is to group scenes into days such that the total duration of scenes scheduled on each day does not exceed 12 hours, while prioritizing scenes with the same location and actors.

  - Objective Function:
    - w1 * pulp.lpSum(x[i, d] * d for i in scenes for d in days): Minimize the total number of days used.
    - w2 * pulp.lpSum(y[location, d] for location in set(scene['location'] for scene in scenes.values()) for d in days): Minimize the number of different locations per day.
    - w3 * pulp.lpSum(delta_location[location, d] for location in set(scene['location'] for scene in scenes.values()) for d in days if d > 1): Minimize the number of location switches between consecutive days.
    - w4 * pulp.lpSum(delta_actors_between_scenes[i, j, d] for i in scenes for j in scenes for d in days if i != j): Minimize the number of actor switches between consecutive scenes within a day.
    - w5 * pulp.lpSum(delta_actor[actor, d] for actor in actors for d in days if d > 1): Minimize the number of actor switches between consecutive days.
  
  - Constraints:
    - Each scene is scheduled exactly once.
    - The total duration of scenes in any day does not exceed 12 hours.
    - Encourage scenes in the same location are assigned to the same day.
    - Encourage scenes have same actors are assigned to the same day.
    - Encourage consecutive days to have the same location.

#### Limitations:
  
- The problem is simplified here, the cost of actors, props, scene settings and many other factors are not considered here.
-	The MILP algorithm developed here is not very efficient and the performance also seems not very good. Need to tune the objective function and constraints to achieve better results.

#### Improvement:
  
  -	Consider more factors such different daily wage of actors, cost of the crew, waiting time for the actors, transfer cost of the scene location, etc.
  -	Develop other methods such as tabu search based method (TSBM) and particle swarm optimization based method (PSOBM) to solve large-scale problems.




