# portcullis
Trading Platform

## Requirements

NOTE: The requirements will be replaced with a dockerized environment in the near future. For now, a local installation of python and creation of a virtual environment are required following the steps outlined below. 

1. Install Python version 3.x (e.g. from https://www.python.org/downloads/ or consult your OS installation manuals.)
- Verify that Python is properly installed, e.g. via ```python --version```
2. NOTE: This step requires a valid SSH key on your system. To create this key follow the instructions on https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent and ensure that the public key is uploaded to your github account for access authentication.
- Clone the repository via SSH, e.g. ```git clone git@github.com:dewrussdx/portcullis.git```. Change into the directory containing the respository, e.g. ```cd portcullis```
2. Create Python virtual environment
- Run ```python -m venv .venv``` to create a virtual environment in the ```.venv``` directory
- Activate the Python virtual environment via ```.venv\Scripts\activate.bat``` (Windows) or ```source ./.venv/bin/activate``` (Linux/MacOS)
3. Install Portcullis Python package dependencies
- ```python -m pip install -r requirements.txt```
4. Run Portcullis via ```python ./main.py```


This framework is using the yfinance package to manage financial information. Documentation can be found here: https://pypi.org/project/yfinance/
