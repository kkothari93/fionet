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

There are three different inverse problems tested in the paper -- **R**everse **T**ime **C**ontinuation (showcases diffeomorphisms in all their glory), **I**nverse **S**ource problem (inspired by photoacoustic imaging) and reflector imaging (inspired by seismic imaging).

## dataset generation

We use the MATLAB kWave toolbox to generate the data. In the `data_gen/` folder we have the kwave scripts used to generate the data. These scripts consume `.png` images that act as source (in RTC, ISP) and as medium perturbation in reflector imaging. These `.png` images are generated via associated python scripts. 

I will try to upload tensorflow records of all training datasets and numpy files for all test datasets in a Box link. This would be free to download. In order to recreate the dataset instead of downloading, follow the steps below.

### reverse time continuation (rtc)

First generate a small dataset of thick lines by running `python3 random_lines_dataset --problem rtc` . This will generate a folder `thick_lines/` within `data_gen/`. Each image in `thick_lines/` is then treated as a source pressure over a fixed Gaussian background wavespeed. The source pressure is propagated for time $T$ to get the final pressure snapshot.

To generate test data from different datasets, convert it into a $512 \times 512$ image and store into a folder. Then run the appropriate submission script and generate test data.

### inverse source problem (isp)

Use `thick_lines/` folder from rtc and run the `kwave_isp.m` script. 

### reflector imaging 

Run `python3 random_lines_dataset --problem refl` from the `data_gen/` folder. This will create a directory of images called `reflectors/`. These will be used as medium background wave speed in kWave.



Work in progress

- Links to datasets or scripts to generate data from the paper will be added. 
- Results will be added.
- help for experiment and data processing pipelines will be added.
