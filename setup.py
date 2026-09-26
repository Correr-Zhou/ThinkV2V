from setuptools import find_packages, setup


setup(
    name="thinkv2v",
    version="0.1.0",
    description="ThinkV2V video editing training, inference, and evaluation code.",
    packages=find_packages(),
    include_package_data=True,
    package_data={"diffsynth": ["tokenizer_configs/**/**/*.*"]},
    python_requires=">=3.10",
)
