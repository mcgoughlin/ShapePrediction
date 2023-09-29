from setuptools import setup,find_packages

setup(
    name='PCA-k',
    packages=find_packages('PCA-k', exclude=['test']),
    install_requires=['matplotlib','numpy','scipy','pandas','open3d',
                      'scikit-learn','tqdm','torch'],
    python_requires='>=3.10',
    description='Python Package for finding the average kidney shapes',
    version='0.1',
    url='https://github.com/mcgoughlin/KCD',
    author='bmcgough',
    author_email='billy.mcgough1@hotmail.com',
    keywords=['pip','shape','PCA']
    )
