from setuptools import find_packages, setup

package_name = 'ncvx_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['./config/obstacles.yaml'])

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eehaap',
    maintainer_email='eemil.haaparanta@tuni.fi',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ncvx_nav_node = ncvx_navigation.ncvx_controller:main'
        ],
    },
)
