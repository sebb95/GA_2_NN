# Snake Reinforcement Learning

Graded assignment 2 Sebastian Tøkje Neural Networks

5 Samples from the best performing agent iteration:

![Trained Agent Policy](images/1.gif)
![Trained Agent Policy](images/2.gif)
![Trained Agent Policy](images/3.gif)
![Trained Agent Policy](images/4.gif)
![Trained Agent Policy](images/5.gif)

Had 2 install 2 dependencies to my Ubuntu 22 LTS to get the code to run before i started editing the project:

Needed to run the video generation.
ffmpeg and libiconv



Train the model:-----------------

    conda activate "your_ga01_env_name"
    python training_pytorch_finish.py



Run/ visualize the model (Generate video)
Finds the best model so far (max mean reward of all in csv file)
And uses it to visualize along with a "bad" start model

Run:----------------

    python game_visualization_pytorch.py



Additional dependencies:
ffmpeg
libiconv



