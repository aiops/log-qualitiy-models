from setuptools import setup

setup(
    name='qulog_svc',
    version='1.0',
    description='Model for log level quality assessment.',
    author='logsight.ai',
    author_email='info@logsight.ai',
    url='https://github.com/aiops/log-qualitiy-models',
    packages=['qulog_svc'],
    include_package_data=True,
    package_data={
        'qulog_svc': ['model'],
    },
    install_requires=[
        "scikit-learn==0.24.2",
        "spacy==3.1.0",
    ],
)
