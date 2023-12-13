Contribution instructions for collaborators
===========================================

.. warning::
    These instructions are for authorized maintainers only


To contribute to this repository, please follow these steps:


**1. Ensure you have the latest version of the repository:**
    
Before you start working on your contribution, it's crucial to have the most up-to-date version of the repository to avoid conflicts with other contributors' changes.

First, navigate to your local repository's directory and then run the following commands to fetch the latest changes from the remote repository and update your local branch:

.. code-block::

    git fetch origin
    git pull origin main  # Replace 'main' with the name of the branch you plan to work off of 

This ensures that your local branch is synchronized with the latest changes from the main branch.

**2. Checkout a new branch for your work:**

Before you start making any changes, it's safe to work in a dedicated branch to keep your changes isolated. This helps in managing multiple contributions simultaneously and keeping the main branch clean.

To create and switch to a new branch with a meaningful name (replace your-branch-name with a descriptive name for your task):

.. code-block::

    git checkout -b your-branch-name

Now you're in your new branch and ready to make changes.


**3. If your changes involve code modifications, run tests to ensure everything is working correctly:**

Running tests before submitting your changes helps ensure that you haven't introduced any new issues.

The tests, found in the tests/ directory, should be run with pytest.

**4. Create a pull request on GitHub:**

When creating a pull request, please describe your changes and the problem they solve.

If your pull request is accepted, it will be merged into the relevant branch.


Thank you and please ask if you have any questions!
