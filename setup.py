from setuptools import setup, find_namespace_packages
import os

setup(name='dotai.trackpingpong.tracknet', 
    version='0.1.0.0.0',
    description='Description for dotai.trackpingpong.tracknet module', 
    url='https://dotai.cloud', 
    author='liwuchen', 
    author_email='liwch@dotai.cloud', 
    license='Apache', 
    packages=find_namespace_packages(),
    scripts=['scripts/'+f for f in os.listdir(os.path.dirname(os.path.realpath(__file__))+'/scripts')],
    python_requires='>=3.6.12',
    install_requires=
    [
        'pyarmor',
        'opencv==4.5.4',
        'focal-loss',
        'numpy==1.21.3',
        'tensorflow==2.6.0',
        'yaml==5.1',
        'pandas==1.3.3'
        
    ],      
    zip_safe=True)
