import setuptools

print(setuptools.find_packages())

setuptools.setup(name="shachtav_ml",
                 version="0.2.0",
                 author="Guy Knaan",
                 author_email="guyknaan@gmail.com",
                 packages=setuptools.find_packages(),
                 install_requires = [
                     "pretty_midi",
                     "music21",
                     "git+https://github.com/CPJKU/madmom.git",
                     "numpy",
                     "tensorflow~=2.15.0",
                     "keras~=2.15.0",
                     "tqdm",
                     "librosa",
                     "pandas"
                 ])
