from setuptools import setup

setup(
    name='level_qulog_sm_rf', # !!! Autogenerated !!! Do not change the name
    version='1.0',
    description='Model for log level quality assessment.',
    author='logsight.ai',
    author_email='info@logsight.ai',
    url='https://github.com/aiops/log-qualitiy-models',
    packages=['level_qulog_sm_rf'], # !!! Autogenerated !!! Do not change the name
    include_package_data=True,
    package_data={
        'level_qulog_sm_rf': ['model'], # !!! Autogenerated !!! Do not change the name
    },
    install_requires=[
        "scikit-learn==0.24.2",
        "spacy==3.1.0",
        "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.1.0/en_core_web_sm-3.1.0-py3-none-any.whl"
    ],
)
