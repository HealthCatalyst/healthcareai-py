# Contributing

You want to help? Woohoo! We welcome that and are willing to help newbies get started.

Please see [our contribution guidelines](https://github.com/HealthCatalyst/healthcareai-py/blob/master/CONTRIBUTING.md) for instructions on setting up your development environment

## Workflow

1. [Identify an issue that](https://github.com/HealthCatalyst/healthcareai-r/issues) suits your skill level
    * Only look for issues in the Backlog category
    * If you're new to open source, please look for issues with the `bug low`, `help wanted`, or `docs` tags
    * Please reach out with questions on details and where to start
2. Create a topic branch to work in; here are [instructions](CONTRIBUTING.md#create-a-topic-branch-that-you-can-work-in)
3. Create a throwaway file on the Desktop (or somewhere outside the repo), based on an example
4. Make changes and use the throwaway file to validate that your packages changes work
    * Make small commits after getting a small piece working
    * Push often so your changes are backed up. See [this](https://gist.github.com/blackfalcon/8428401#push-your-branch) for more details.
5. Early on, create a [pull request](https://yangsu.github.io/pull-request-tutorial/) such that Levi and team can discuss the changes that you're making. Conversation is good.
6. When you have resolved the issue you chose, do the following:
    * Check that the unit tests are passing
    * Check that pyflakes and pylint don't show any issues
    * Merge the master branch into your topic branch (so that you have the latest changes from master)
        ```bash
        git checkout LeviBugFix
        git fetch
        git merge --no-ff origin/master
        ```
    * Again, check that the unit tests are passing
7. Now that your changes are working, communicate that to Levi in the pull request, such that he knows to do the code
  review associated with the PR. Please *don't* do tons of work and *then* start a PR. Early is good.


## How to set up your dev environment

### Set up git

#### Windows

- [Download and install](https://git-scm.com/download/) git for Windows
- Choose "64-bit Git for Windows Setup" (assuming you're on modern hardware)
- On the Select Components screen, accept the defaults
- After selecting install location, choose "Use Git from the Windows Command Prompt"
- Checkout using Windows-style
- Choose MinTTy

#### macOS

see [macOS instructions](https://developer.apple.com/xcode/) for macOS and 

#### Linux

[linux instructions](https://git-scm.com/download/linux)

### Install Python 3.5

#### Windows

- Download [Anaconda for Windows](https://www.continuum.io/downloads) (Python 3.5) 

#### macOS

There are a number of ways to install python 3.5 on macOS. Here are a few different ways:

- If you already have [homebrew](http://brew.sh), (a package manager for macOS), you can simply run `brew install python3`
- You can also install with the excellent [Anaconda distribution](https://www.continuum.io/downloads)
- You can also follow the [Hitchhiker's Guide to Python](http://docs.python-guide.org/en/latest/starting/install/osx/) instructions if you prefer a more hands-on approach.

#### Linux

- Use your package manager (apt-get, yum, etc), for example `sudo apt-get install python3`

### Required packages

- Open the terminal (i.e., git bash, if using Windows)
- run `conda install pyodbc`

## Set up healthcare-ai

Install directly from [our repo](https://github.com/HealthCatalyst/healthcareai-py)

- run `pip install https://github.com/HealthCatalyst/healthcareai-py/zipball/master`
- Fork this repo (look for the link in the top right corner of the current page)
- At the top of the [repo homepage](https://github.com/HealthCatalyst/healthcareai-py) click the green 'Clone or download' button and copy the https link
- In your terminal run `git clone <PASTE THE LINK HERE>`
- `cd` healthcareai-py directory

If you like using virtual environments (not required):

- In terminal, run `conda env create` to create the hcconda virtual environment
- To activate your virtual environment, in terminal run `activate hcconda` (or `source activate hcconda` if using bash)
- You might have to update packages, especially sklearn. The best way to do this is through Settings->Project->Project Interpreter and update scikit-learn

## Install an IDE and clone the healthcareai-py repo

### Configuring PyCharm

1. Install PyCharm Community Edition
2. Set path to git.exe via File -> settings -> Version Control -> Git
3. Clone repo (if you haven't) via PyCharm -> VCS (at top) -> Checkout from version control -> Github
    - Grab .git url from healthcareai-py repo in Github
4. File -> Open and look for the healthcareai project (if it didn’t come up already)

### SQL Server

**Note that there are only a few true integration tests that use MSSQL server as a destination.** If you are not interested in running these few tests, feel free to skip this section.

If on Windows, [install both](http://stackoverflow.com/a/11278818/5636012_SQL) Server Express and SSMS Express
- Navigate to the [downloads page](https://www.microsoft.com/en-us/download/details.aspx?id=29062)
- Look for and download **ENU\x64\SQLEXPRWT_x64_ENU.exe**
- When installing, be sure to check the box to install SSMS

### Create some tables in your database

- Create tables on localhost

Note that these will go in the SAM database, if using the Health Catalyst analytics environment

        ```sql
        CREATE TABLE [SAM].[dbo].[HCAIPredictionClassificationBASE] (
        [BindingID] [int] ,
        [BindingNM] [varchar] (255),
        [LastLoadDTS] [datetime2] (7),
        [PatientEncounterID] [decimal] (38, 0),
        [PredictedProbNBR] [decimal] (38, 2),
        [Factor1TXT] [varchar] (255),
        [Factor2TXT] [varchar] (255),
        [Factor3TXT] [varchar] (255))
        
        CREATE TABLE [SAM].[dbo].[HCAIPredictionRegressionBASE] (
        [BindingID] [int],
        [BindingNM] [varchar] (255),
        [LastLoadDTS] [datetime2] (7),
        [PatientEncounterID] [decimal] (38, 0),
        [PredictedValueNBR] [decimal] (38, 2),
        [Factor1TXT] [varchar] (255),
        [Factor2TXT] [varchar] (255),
        [Factor3TXT] [varchar] (255))
        ```

#### Configuration of localhost alias for MSSQL

You will need to have an alias called "localhost" that points to your SQL database.

1. Open SQL Server Configuration Manager
   - On the left pane, expand SQL Native Client 11.0 Configuration and right click Aliases. Select New Alias.
   - In the pop up dialog box, enter the following and then press OK:
     - Alias Name: localhost
     - Port No: 1433
     - Protocol: TCP/IP
     - Server: YOUR_COMPUTER_NAME (this should be something like HC2080, if you can connect to HC2080 in SSMS)
2. On the left pane, expand SQL Server Network Configuration and click on Protocols for SQLEXPRESS.
   - Right click on TCP/IP and click Properties. In the dialog box, enter the following and then press OK:
     - Under the Protocol tab, verify that Enabled is set to Yes.
     - Under the IP Addresses tab, scroll all the way to the bottom.
     - Under IPALL, TCP Dynamic Ports should be blank
     - Under IPALL, TCP Port should be set to 1433
3. Working in Configuration Manager,
   - Expand SQL Server Services
   - Right-click on your SQL Server instance
   - Restart SQL Server
4. Open SSMS and verify that you can connect to the server `localhost`.

### Verify that unit tests pass

- Right click on tests folder under healthcareai-py/healthcareai
- Click on Run Nosetest in test

### Create test branch and push it to github

- Note the text ‘Git: master’ in bottom-right of PyCharm
- Create new test branch via VCS -> Git -> Branches -> New Branch
- Push branch to github (ie, create origin) via VCS -> Git -> Push (CTRL-SHIFT-K)

## Code Housekeeping

1.  Install the following packages via the command line: `python -m pip install packagename`
  + pylint
  + pyflakes
2. Set these up via http://www.mantidproject.org/How_to_run_Pylint#PyCharm_-_JetBrains
  + If your python is installed in `C:\Pythonxx`, then your parameters will be:
  + Program: `C:\Python34\Scripts\pylint.exe`
  + Parameters: `$FilePath$`
  + Working dir: `C:\Python34\Scripts`
  + If you are using a different Python distribution, you may need to find where Pylint is installed.  For example, the same three parameters from above might be:
  + `C:\Users\user.name\AppData\Local\Continuum\Anaconda3\Scripts\pylint`
  + Parameters: `$FilePath$`
  + `C:\Users\user.name\AppData\Local\Continuum\Anaconda3\Scripts`
  + Instead of using default parameter, use `$FilePath$`
  + For Anaconda, you may have to use `C:\Users\user.name\AppData\Local\Continuum\Anaconda3\Scripts\pylint`
  + Check all boxes
3. Make sure pylint and pyflakes work
  + Right-click on relevant directory in PyCharm (this will be where you’ve done work)
  + Navigate to external tools
  + Run both pylint and pyflakes
  + Verify that there aren’t any issues with your code; please do this before sending pull requests

4. Set maximum line width to 79 via Settings -> Editor -> Code Style -> Right margin
5. Set tabs as spaces via Edit -> Convert Indents -> To Spaces
6. Click Code -> Inspect code -> Whole project -> Look for section on Package requirements
  + Under the lines related to sklearn, click ‘Ignore Requirement’

## Git config

Set up your email and username for git (otherwise no attribution in github)

1. Open a terminal (ie, git bash, if on Windows)
2. Set up your email and user name for proper attribution
  + `git config user.name "Billy Everyteen"`
  + `git config --global user.email "your_email@example.com"`
3. Configure line endings for windows: `git config core.autocrlf true`
4. Make git case sensitive for file names: `git config core.ignorecase false`
5. Improve merge conflict resolution via `git config --global merge.conflictstyle diff3`
6. If you use a personal email for github, and would rather have notifications go to your Health Cataylst email
- See [GitHub Notification Settings](https://github.com/settings/notifications) -> Notification email -> Custom routing
7. Set up SSH (if desired) so you can push to topic branch without password
  + [Step1](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/)
  + [Step2](https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account/)
  + [Step3](https://help.github.com/enterprise/11.10.340/user/articles/changing-a-remote-s-url/)

## Create a topic branch that you can work in

1. Open the terminal (whether in Git Bash, in Spyder, etc)
2. Type `git checkout -b nameofbranch`
  + This creates the your local branch for work
  + Note this branch should have your name in it
3. Type `git push origin nameofbranch`
  + This pushes the branch to github (and now it's backed-up)

## Getting started with contributions

When your dev environment is set up, see [the contribution workflow](README.md#contributing).
