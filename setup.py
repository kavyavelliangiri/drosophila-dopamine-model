from setuptools import setup, find_packages

setup(name='mbmodel',
      version='1.0.0',
      description="A computational model of dopaminergic learning and associative olfactory memory in Drosophila",
      author='Kavya Velliangiri',
      packages=find_packages(where='src'),
        package_dir={'': 'src'},
        python_requires='>=3.7',
        install_requires=[
            'numpy',
            'scipy',
            'matplotlib',
            'pandas',
            'pyyaml',
        ],
        extras_require={
            'dev': ['pytest']
        }
)