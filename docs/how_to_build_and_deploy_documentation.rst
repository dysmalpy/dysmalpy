How to build and deploy documentation
============

**NOTE: These instructions are for authorized maintainers only**

To update the documentation of dysmalpy, which is located at the live server [https://www.mpe.mpg.de/resources/IR/DYSMALPY/](https://www.mpe.mpg.de/resources/IR/DYSMALPY/), please follow these steps:

**1. Make changes ONLY in the dysmalpy repository**

To have an organized workflow and leverage from version control, we will make the changes on the documentation (e.g., on the .rst files) in the github repository and not directly in the AFS directories.

Refer to the file ```workflow_on_how_to_contribute.rst``` on how to contribute to the repository. 

**2. Pull the changes to the AFS directory**

When the repository is updated, you can pull those changes to the astrosoft directory. Start an ssh session and navigate to the directory in /afs/:

.. code-block:: console

    $ cd /afs/mpe/astrosoft/dysmalpy/dysmalpy/docs

Pull the changes from the repository:

.. code-block:: console

    $ git pull

(Make sure you have the relevant access rights to the /afs/ directory. If you don't, please contact Thomas Ott.)

**3. Build the documentation**

To build the documentation with Sphinx, which compiles the files in reStructuredText format (reST) with extension .rst  and produces the .html files, run the following command:

.. code-block:: console

    $ make html

The ```Makefle``` document that is executed using the command above has the important paths and the instrucitons for Sphinx so it should not be changed, unless strictly necessary. The built .html docs will go under ```/afs/mpe/astrosoft/dysmalpy/dysmalpy/docs/_build/html```

**4. Copy the built docs to the live server**

To automate the process of copying the .html files to the live website using rsync, run the following shell script, ```update.sh```:

.. code-block:: console

    $ bash /afs/mpe/www/resources/IR/DYSMALPY/update.sh

The .html files will be copied to ```/afs/mpe/www/resources/IR/DYSMALPY/```, so the website will be updated automatically. Always double-ckeck that the website is updated correctly.

Thank you for contributing to Dysmalpy's documentation. If you have any questions or encounter issues during the process, feel free to ask for assistance.