import setuptools

print(setuptools.find_packages())

setuptools.setup(name="shachtav_ml",
                 version="0.2.0",
                 author="Guy Knaan",
                 author_email="guyknaan@gmail.com",
                 packages=setuptools.find_packages())