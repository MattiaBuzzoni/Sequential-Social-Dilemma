from setuptools import setup

setup(name='gym-social-dilemma',
      author='Mattia Buzzoni',
      version='0.1.1',
      description='This is an OpenAI gym an implementation and extension of the Commons Game, a multi-agent '
                  'environment proposed in Multi-agent Reinforcement Learning in Sequential Social Dilemma '
                  'using pycolab as game engine.',
      install_requires=['gym', 'pycolab', 'matplotlib', 'numpy', 'scipy'],
      python_requires='>=3.11',
      )
