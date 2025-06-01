# Headline-Generation-Project
Steps to run the training function on the terminal:
1) Navigate to the Headline-Generation-Project folder after cloning the git repository
2) Run the following line in the terminal: python -m src.train. Depending on the model used, use the appropriate function imported from model_loader.
3) After training the BART/Pegasus model, a safe tensors file is generated. Add the respective files to the respective bart/pegasus models in the models folder.
4) Update the directories in the generate and evaluate files by copy pasting the path of the bart/pegasus folder depending on the model being used to generate/evaluate words
5) To run generate and evaluate functions run python -m src.generate or python -m src.evaluate
