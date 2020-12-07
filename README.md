# FIONet
Official repo for "Learning the geometry of wave-based imaging" [arxiv link](https://arxiv.org/abs/2006.05854)

We propose a physics-based learning solution to wave-based imaging problems that have applications ranging from biomedical to seismic imaging. This repository is intended to provide code to reproduce the results from the paper. This is a work-in-progress.

The key idea of this work is using the fact that waves bend when traveling in heterogeneous media. We separate learning the wave physics into learning the bending in space and time. 

Bibtex
~~~bib
@article{kothari2020learning,
  title={Learning the geometry of wave-based imaging},
  author={Kothari, Konik and de Hoop, Maarten and Dokmani{\'c}, Ivan},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
~~~  

There are three different inverse problems tested in the paper -- Reverse Time continuation (showcases diffeomorphisms in all their glory), inverse source problem (inspired by photoacoustic imaging) and reflector imaging (inspired by seismic imaging).

## dataset generation

We use the MATLAB kWave toolbox to generate the data. In the `data_gen/` folder we have the kwave scripts used to generate the data. These scripts consume `.png` images that act as source (in )




Work in progress

- Links to datasets or scripts to generate data from the paper will be added. 
- Results will be added.
- help for experiment and data processing pipelines will be added.
