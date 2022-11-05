---
title: Anaconda2和3的并存一键安装
tags:
  - python
  - anaconda
date: 2018-08-13 19:55:48
---


首先贴上Conda官网教程的[网址](https://conda.io/docs/user-guide/tasks/manage-python.html)

确实还是官网教程最简单实用，许多百度甚至谷歌搜索出来的csdn博客等并不是最优解法，而且废话太多...

有效解决办法就是下面一句（已经装了Anaconda3想装一个2）：

```bash
conda create -n py27 python=2.7 anaconda
```

或是这一句（已经装了Anaconda2想装一个3）：

```bash
conda create -n py36 python=3.6 anaconda
```

# Managing Python

- [Viewing a list of available Python versions](https://conda.io/docs/user-guide/tasks/manage-python.html#viewing-a-list-of-available-python-versions)
- [Installing a different version of Python](https://conda.io/docs/user-guide/tasks/manage-python.html#installing-a-different-version-of-python)
- [Using a different version of Python](https://conda.io/docs/user-guide/tasks/manage-python.html#using-a-different-version-of-python)
- [Updating or upgrading Python](https://conda.io/docs/user-guide/tasks/manage-python.html#updating-or-upgrading-python)

Conda treats Python the same as any other package, so it is easy to manage and update multiple installations.

Anaconda supports Python 2.7, 3.4, 3.5 and 3.6. The default is Python 2.7 or 3.6, depending on which installer you used:

- For the installers “Anaconda” and “Miniconda,” the default is 2.7.
- For the installers “Anaconda3” or “Miniconda3,” the default is 3.6.

## [Viewing a list of available Python versions](https://conda.io/docs/user-guide/tasks/manage-python.html#id1)

To list the versions of Python that are available to install, in your Terminal window or an Anaconda Prompt, run:

```
conda search python
```

This lists all packages whose names contain the text `python`.

To list only the packages whose full name is exactly `python`, add the `--full-name` option. In your Terminal window or an Anaconda Prompt, run:

```
conda search --full-name python
```

## [Installing a different version of Python](https://conda.io/docs/user-guide/tasks/manage-python.html#id2)

To install a different version of Python without overwriting the current version, create a new environment and install the second Python version into it:

1. Create the new environment:

   - To create the new environment for Python 3.6, in your Terminal window or an Anaconda Prompt, run:

     ```
     conda create -n py36 python=3.6 anaconda
     ```

     NOTE: Replace `py36` with the name of the environment you want to create. `anaconda` is the metapackage that includes all of the Python packages comprising the Anaconda distribution. `python=3.6` is the package and version you want to install in this new environment. This could be any package, such as `numpy=1.7`, or [multiple packages](https://conda.io/docs/user-guide/tasks/manage-pkgs.html#installing-multiple-packages).

   - To create the new environment for Python 2.7, in your Terminal window or an Anaconda Prompt, run:

     ```
     conda create -n py27 python=2.7 anaconda
     ```

2. [Activate the new environment](https://conda.io/docs/user-guide/tasks/manage-environments.html#activate-env).

3. Verify that the new environment is your [current environment](https://conda.io/docs/user-guide/tasks/manage-environments.html#determine-current-env).

4. To verify that the current environment uses the new Python version, in your Terminal window or an Anaconda Prompt, run:

   ```
   python --version
   ```

## [Using a different version of Python](https://conda.io/docs/user-guide/tasks/manage-python.html#id3)

To switch to an environment that has different version of Python, [activate the environment](https://conda.io/docs/user-guide/tasks/manage-environments.html#activate-env).

## [Updating or upgrading Python](https://conda.io/docs/user-guide/tasks/manage-python.html#id4)

Use the Terminal or an Anaconda Prompt for the following steps.

If you are in an environment with Python version 3.4.2, the following command updates Python to the latest version in the 3.4 branch:

```
conda update python
```

The following command upgrades Python to another branch—3.6—by installing that version of Python:

```
conda install python=3.6
```