Building and deploying documentation
====================================

.. warning::
    These instructions are for authorized maintainers only. Note that you need to 
    have the C++ extensions in order to build the documentation without any errors.

Building documentation locally
******************************

Before committing documentation changes, it is recommended to check the changes 
by building the documentation on your computer. 

**1. Install Sphinx and dependencies**

You can install Sphinx and the dependencies to build the documentation using pip:

.. code-block::

    pip install sphinx-automodapi sphinx-rtd-theme myst_nb sphinx-copybutton

**2. Make changes within your working branch**

For both documentation and code changes, all work should be done in a new 
branch off of the main or development branch (as appropriate). 

**3. Build the documentation**

Once you have made changes on any of the documentation files under /docs/,
you can build the documentation with Sphinx, which compiles the files in 
reStructuredText format (reST) with extension .rst and Jupyter notebook 
tutorials (.ipynb)  and produces the .html files, run the following command:

.. code-block::

    make html

The `Makefle` document that is executed using the command above has the 
important paths and the instructions for Sphinx so it should not be changed, 
unless strictly necessary. 

**4. Check built html files**

The built .html docs are located under 
`/PATH/TO/YOUR/DYSMALPY/INSTALL/dysmalpy/docs/_build/html/`

Within your browser, open the local file 
`/PATH/TO/YOUR/DYSMALPY/INSTALL/dysmalpy/docs/_build/html/index.html` 
and check the various documentation pages you have changed/added, including 
any added links to the index page TOC. If you are using VScode, you can install 
the extension "Live Server" to view the built html files in your browser.

**5. Iterate as necessary**

Commit the changes after all testing and work are completed.  


Building and deploying to the live server
*****************************************

To update the documentation of dysmalpy, which is located at the live server 
https://www.mpe.mpg.de/resources/IR/DYSMALPY/, please follow these steps:

**1. Make changes ONLY in the dysmalpy repository**

To have an organized workflow and leveraged version control, we will make 
the changes on the documentation (e.g., on the .rst files) in the GitHub 
repository and not directly in the AFS directories.

Refer to the file `workflow_on_how_to_contribute.rst` on how to contribute 
to the repository. 

**2. Pull the changes to the AFS directory**

When the repository is updated, you can pull those changes to the astrosoft 
directory. Start an ssh session, then activate the dysmalpy environment with:

.. code-block::

    source /afs/mpe/astrosoft/dysmalpy/dysmalpy_setup.sh

Then, navigate to the directory in /afs/:

.. code-block::

    cd /afs/mpe/astrosoft/dysmalpy/dysmalpy/docs

Pull the changes from the repository:

.. code-block::

    git pull

(Make sure you have the relevant access rights to the /afs/ directory. If you 
don't, please contact Thomas Ott.)

**3. Build the documentation**

To build the documentation with Sphinx, which compiles the files in 
reStructuredText format (reST) with extension .rst and Jupyter notebook 
tutorials (.ipynb) and produces the .html files, run the following command:

.. code-block::

    make html

The `Makefile` document that is executed using the command above has the 
important paths and the instructions for Sphinx so it should not be changed, 
unless strictly necessary. The built .html docs will go under 
`/afs/mpe/astrosoft/dysmalpy/dysmalpy/docs/_build/html`

**4. Copy the built docs to the live server**

To automate the process of copying the .html files to the live website using 
rsync, run the following shell script, ``update.sh``:

.. code-block::

    bash /afs/mpe/www/resources/IR/DYSMALPY/update.sh

The .html files will be copied to `/afs/mpe/www/resources/IR/DYSMALPY/`, so 
the website will be updated automatically. Always double-check that the website 
is updated correctly.

Thank you for contributing to Dysmalpy's documentation. If you have any 
questions or encounter issues during the process, feel free to ask for 
assistance.
