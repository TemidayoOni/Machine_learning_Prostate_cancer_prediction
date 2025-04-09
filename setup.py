from setuptools import setup, find_packages
from typing import List


# function to get requirements from the requirements.txt file
EDOT = '-e .'
def get_requirements(file_path:str)->List[str]:

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [r.replace("\n", "") for r in requirements]

        if EDOT in requirements:
            requirements.remove(EDOT)
    
    return requirements



setup(
    name = "ml prostate cancer",
    version="0.0.1",
    author="Temidayo Oni",
    author_email="onitemdayo@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)