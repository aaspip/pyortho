**Pyortho**
======
#  [Random noise attenuation using local signal-and-noise orthogonalization](https://www.researchgate.net/publication/270902763_Random_noise_attenuation_using_local_signal-and-noise_orthogonalization)

![GitHub last commit](https://img.shields.io/github/last-commit/aaspip/pyortho?style=plastic)
![Twitter Follow](https://img.shields.io/twitter/follow/aaspip?style=social)
![GitHub followers](https://img.shields.io/github/followers/aaspip?style=social)
![GitHub stars](https://img.shields.io/github/stars/aaspip/pyortho?style=social)
![GitHub forks](https://img.shields.io/github/forks/aaspip/pyortho?style=social)

## Description

**Pyortho** is a python package for local signal-and-noise orthogonalization and local similarity calculation. The local orthogonalization method is a fundamental seismic data analysis algorithm and has a wide range of applications. More examples will be continuously updated. 

NOTE:
If you use this package to calculate local orthogonalization weight and local similarity, please show courtesy to cite the reference below. Using local similarity to calculate the signal leakage was first ever used in the work by Chen and Fomel (2015), please show courtesy if you use it for the same purpose.

0. [Install](#Install)
0. [Examples](#Examples)
0. [Dependence Packages](#Dependence_Packages)
0. [Development](#Development)
0. [Contact](#Contact)
0. [Gallery](#Gallery)

## Reference

    Chen, Y. and S. Fomel. "Random noise attenuation using local signal-and-noise orthogonalization." Geophysics 80, no. 6 (2015): WD1-WD9.

BibTeX:

	@article{chen2015random,
	  title={Random noise attenuation using local signal-and-noise orthogonalization},
	  author={Chen, Yangkang and Fomel, Sergey},
	  journal={Geophysics},
	  volume={80},
	  number={6},
	  pages={WD1--WD9},
	  year={2015},
	  publisher={Society of Exploration Geophysicists}
	}

-----------
## Copyright
	pyortho developing team, 2022-present
-----------

## License
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)   

-----------

## Install
Using the latest version

    git clone https://github.com/aaspip/pyortho
    cd pyortho
    pip install -v -e .
or using Pypi

    pip install pyortho

-----------
## Examples
    The "demo" directory contains all runable scripts to demonstrate different applications of pyortho. 

-----------
## Dependence Packages
* scipy 
* numpy 
* matplotlib

-----------
## Development
    The development team welcomes voluntary contributions from any open-source enthusiast. 
    If you want to make contribution to this project, feel free to contact the development team. 

-----------
## Contact
    Regarding any questions, bugs, developments, collaborations, please contact  
    Yangkang Chen
    chenyk2016@gmail.com

-----------
## Gallery
The gallery figures of the MATortho package can be found at
    https://github.com/chenyk1990/gallery/tree/main/matortho
Each figure in the gallery directory corresponds to a DEMO script in the "demo" directory with the exactly the same file name.

The following figure shows a 2D denoising example using the pyortho package. Generated by [demos/test_pyortho_localortho2d.py](https://github.com/aaspip/pyortho/tree/main/demos/test_pyortho_localortho2d.py)
<img src='https://github.com/chenyk1990/gallery/blob/main/matortho/test_localortho2d.png' alt='comp' width=960/>

The following figure shows a 3D denoising example using the pyortho package. Generated by [demos/test_pyortho_localortho3d.py](https://github.com/aaspip/pyortho/tree/main/demos/test_pyortho_localortho3d.py)
<img src='https://github.com/chenyk1990/gallery/blob/main/matortho/test_localortho3d.png' alt='comp' width=960/>






