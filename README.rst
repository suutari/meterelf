Meter Elf
=========

This project implements dial meter value reading by computer vision
using OpenCV with Python.

It currently works just for my configuration with exactly the same
parameters (like type of water meter, position and angle of the webcam,
lighting conditions, etc.), but I plan to make it more general in the
future.

Windows 10 Python environment
-----------------------------
*Anaconda* is open-source package management system for Python. This is a powerful tool that allows users to install packages easily and without root access. It also allows for simple control over virtual environments (called ``conda envs``) that make dependency resolution for particular software or pipelines a breeze.

*Miniconda* is a free minimal installer for conda. It is a small, bootstrap version of Anaconda that includes only conda, Python, the packages they depend on, and a small number of other useful packages, including pip, zlib and a few others. Use the conda install command to install 720+ additional conda packages from the Anaconda repository.

To create needed Python environment install
`miniconda <https://docs.conda.io/en/latest/miniconda.html#windows-installers/>`_. and create conda environment:

.. code:: shell-session

    conda update conda
    conda clean --all
    conda create -n meterelf38 python=3.8
    conda activate meterelf38

Install dependencies:

.. code:: shell-session

    pip install -r requirements-conda.txt

To run the code:

.. code:: shell-session

    (meterelf38) PS C:\Users\..\meterelf> python -m meterelf
    Usage: C:\Users\..\meterelf\meterelf\__main__.py PARAMETERS_FILE [IMAGE_FILE...]

License
-------

All content here is licensed with MIT license.  See the LICENSE file for
details.
