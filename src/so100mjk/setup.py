from setuptools import find_packages, setup
import os 
from glob import glob

package_name = 'so100mjk'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')+ glob('launch/*.py')),
        (os.path.join('share', package_name, 'bag_files'), glob('bag_files/*.db3') + glob('bag_files/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='akshay',
    maintainer_email='aakshay1114@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mjklaunch = so100mjk.mujoco_node:main',
            'ik_final = so100mjk.lm_ik_parallel:main',
            'calibrate = so100mjk.calibrate:main',
        ],
    },
)
