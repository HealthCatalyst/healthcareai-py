# Compiling Python files to an exe

## Python 3.6 Note

Please note that as of 2017-03-31 this process will not work if the machine is running Python 3.6.

## Why might I want to do this?

Note: this is optional to model deployment. When working in with
production environments, occasionally there are machines that one has to
keep relatively pure. In other words, perhaps you don't want to install
a Python (or more specifically, a Python interpreter) on that machine.

But what if you want to run a Python-based predictive model on that same
machine, and tie it into nightly ETL processes? That's why you'd want to
compile your .py file (which runs your model, for example) to an exe--if
you're using Windows--such that it's the only thing you place on the
machine that you want to keep pure. If you're using a OS X, one can
[compile to an .app file](https://pythonhosted.org/py2app/).

## How does this relate to healthcare.ai?

After you've been able to [deploy](http://healthcareai-py.readthedocs.io/en/latest/deploy)
predictions from your model back to SQL Server, you may want to compile
your .py file to an exe.

## The workflow of saving and compiling a model to exe

1)  Train and save your model by running your
    [deploy](http://healthcareai-py.readthedocs.io/en/latest/deploy) .py file with the
    `use_saved_model=False` argument, such that two pkl files are
    created
2)  checked that you can run the model from from the pkl files, by
    setting `use_saved_model=True` and running your .py script again
3)  Note that you should see more rows pushed to SQL Server for 1) and
    2). If you didn't see new rows, something is wrong--fix your
    deploy.py file before proceeding.
4)  If you did see new rows inserted into the database, leave
    `use_saved_model=True` in your script and do the following in
    PowerShell

    -   Install pyinstaller via
        `conda install -c acellera pyinstaller=3.2.3`
    -   Then run

```powershell
pyinstaller.exe --noconfirm --log-level=WARN --clean --nowindow --hidden-import=sklearn.tree._utils --name name_of_executable "C:\Users\name\deploy_script.py" --distpath "C:\Users\name\compiled_model_folder" 
# Copy pkl files (that represent the model) into exe directory just created
xcopy "C:\Users\name\working_directory"*.pkl "C:\Users\name\compiled_model_folder" /Y
```

