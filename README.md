# FaceSimNet
> "Morpheus: Unfortunately no one can be told what the Matrix is. You have to see it for yourself."

## Overview on the project workflow 
"FaceSimNet" is the name I've given to this model, designed to predict the degree of similarity between two input faces. At first glance, this project presents a significant challenge due to the nuanced definition of similarity and the complexity of labeling the data. The pivotal breakthrough came with the realization that the project could be reframed as a face verification task. This simplification was instrumental in shaping the approach to modeling and solving the problem.

## How you should read Notebooks
This project follows a structured workflow that is designed to help you understand the key components and implementation details. Follow the steps below to get started:

1. **Main Notebook**: Begin by reviewing the `main` notebook. This notebook contains theoretical details and serves as an overview or project review. It provides essential background information on the project.

2. **Imgprocessing Notebook**: After reviewing the `main` notebook, proceed to the `Imgprocessing` notebook. This notebook covers various image processing techniques used in the project. 

3. **Model Notebook**: Finally, explore the `model` notebook. This notebook delves into the main ideas and implementation details of the project. By this point, you should have a solid understanding of the project's theoretical foundations and image processing techniques, which will aid in comprehending the model implementation.

By following this structured workflow, you'll gain a comprehensive understanding of the project from its theoretical underpinnings to its practical implementation.

## References 

Research papers: 
- [DeepFace](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)
- [FaceNet](https://arxiv.org/abs/1503.03832)
- [Siamese Network TL](https://ieeexplore.ieee.org/document/9116915)

PyTorch Documentation:
- [Siamese Network Example](https://github.com/pytorch/examples/blob/main/siamese_network/main.py) 
