# ML4BORG
AquilaSpringMeeting20222

### Link to hedgedoc
https://hedgedoc.aquila-consortium.org/YhOWdfjATxicU-fq_fxDJQ?both 

### Installation

* Using ```conda``` execute: 
```
conda env create -f environment.yml
```
* Requirements: ```aquila_borg```,```nbodykit```, ```tensorflow``` and ```pytorch```
* Gotchas:
    - There are dependencies conflicts between **aquila_borg** and **nbodykit**, causing conda to be unable to install **nbodykit**
    - When using training data from L-Gadget **nbodykit** needs to be modified to match the headers 
* Quickfix:
    - Install nbodykit via pip
      
      - need to install ```gcc``` in the conda environment for some specific packages: 
      ```
      conda install -c conda-forge gcc_linux-64 gxx_linux-64
      ```
      
      - pip will install some packages that nbodykit depends on e.g. corrfunc, pfft-python. However, pip is blind some environment modules like **gcc** or **mpicc**. It will assume them to be installed system-wide. Therefore, either install the packages in question globally on your system or (**RECOMMENDED**) install the dependecies via **conda**.
      - For instance one can install the ```pfft-python``` package via 
      ```
      conda install -c bccp pfft-python
      ```
