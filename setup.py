from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as reqs:
        return reqs.read().splitlines()

setup(
    name='piaa',
    version='0.0.1',
    author='Kyungsu Kim',
    author_email='kyungsu.kim@snu.ac.kr',
    description='PIAA for Personal Aesthetic Preference Optimization.',
    packages=find_packages(),
    install_requires=read_requirements(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)