# WeaNF: Weak Supervision with Normalizing Flows

Authors: Andreas Stephan, Benjamin Roth

This repo contains the code related to the Repl4NLP workshop paper

Abstract: 

A popular approach to decrease the need for costly manual annotation of large data sets is weak supervision, which introduces prob- lems of noisy labels, coverage and bias. Meth- ods for overcoming these problems have ei- ther relied on discriminative models, trained with cost functions specific to weak supervi- sion, and more recently, generative models, trying to model the output of the automatic annotation process. In this work, we explore a novel direction of generative modeling for weak supervision: Instead of modeling the output of the annotation process (the labeling function matches), we generatively model the input-side data distributions (the feature space) covered by labeling functions. Specifically, we estimate a density for each weak labeling source, or labeling function, by using normal- izing flows. An integral part of our method is the flow-based modeling of multiple simul- taneously matching labeling functions, and therefore phenomena such as labeling function overlap and correlations are captured. We an- alyze the effectiveness and modeling capabil- ities on various commonly used weak super- vision data sets, and show that weakly super- vised normalizing flows compare favorably to standard weak supervision baselines.



Done:
- baselines/
- layers/


-----

## Data 

Go to github.com/knodle/knodle and download one of the corresponding files.
In scripts/preprocess you'll find code to preprocess and transform data into the needed format.

## Notebooks

We provide notebook which show you how to use our code. 

### Scripts

- Baslines
- Runs. Follow same scheme

## Citation

## Acknowledgements

This research was funded by the WWTF through the project ”Knowledge-infused Deep Learning for Natural Language Processing” (WWTF Vienna Re- search Group VRG19-008), and by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - RO 5127/2-1.

## TODO

- Write doc strings
- Write README
- TODO: Ben: 

