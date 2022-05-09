# WeaNF: Weak Supervision with Normalizing Flows

Authors: Andreas Stephan, Benjamin Roth

This repo contains the code related to the paper [WeaNF: Weak Supervision with Normalizing Flows](https://arxiv.org/abs/2204.13409).
<br />It was accepted at the REPL4NLP workshop co-located at the ACL 2022. 

If there's questions, please contact us [here](mailto:andreas.stephan@univie.ac.at)

---------

## Abstract

A popular approach to decrease the need for costly manual annotation of large data sets is weak supervision, 
which introduces problems of noisy labels, coverage and bias. Methods for overcoming these problems have either relied 
on discriminative models, trained with cost functions specific to weak supervision, 
and more recently, generative models, trying to model the output of the automatic annotation process. 
In this work, we explore a novel direction of generative modeling for weak supervision: 
Instead of modeling the output of the annotation process (the labeling function matches), 
we generatively model the input-side data distributions (the feature space) covered by labeling functions. 
Specifically, we estimate a density for each weak labeling source, or labeling function, by using normalizing flows. 
An integral part of our method is the flow-based modeling of multiple simultaneously matching labeling functions, 
and therefore phenomena such as labeling function overlap and correlations are captured. 
We analyze the effectiveness and modeling capabilities on various commonly used weak supervision data sets, 
and show that weakly supervised normalizing flows compare favorably to standard weak supervision baselines.



-----

## Data 

All datasets are integrated in the [Knodle](github.com/knodle/knodle) framework, and there you can download them.
In scripts/preprocess you'll find code to preprocess and transform data into the needed format.

-----

## Notebooks

In the folder ```notebooks/``` you can find an easy example.

-----

### Scripts

In the ```scripts/``` folder, you find code to reproduce our results.

-----

## Citation

```
@misc{https://doi.org/10.48550/arxiv.2204.13409,
  doi = {10.48550/ARXIV.2204.13409},
  url = {https://arxiv.org/abs/2204.13409},
  author = {Stephan, Andreas and Roth, Benjamin},
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {WeaNF: Weak Supervision with Normalizing Flows},  
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

```

## Acknowledgements

This research was funded by the WWTF through the project ”Knowledge-infused Deep Learning for Natural Language Processing” (WWTF Vienna Re- search Group VRG19-008), and by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - RO 5127/2-1.

