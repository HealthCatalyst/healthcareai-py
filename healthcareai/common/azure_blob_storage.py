def save_output_to_azure_storage(filename,output):
    # Still have to test this.
    # This is from: https://docs.microsoft.com/en-us/azure/storage/storage-python-how-to-use-file-storage
    from azure.storage.file import FileService
    file_service = FileService(account_name=os.environ.get("CAFE_AZURE_FileStorage_ModelLogs_AccountName"), account_key=os.environ.get("CAFE_AZURE_FileStorage_ModelLogs_AccountKey"))

    file_service.create_file_from_path(
        'modellogs',
        None, # We want to create this blob in the root directory, so we specify None for the directory_name
        filename + '.txt')

    file_service.create_file_from_path(
        'modellogs',
        None, # We want to create this blob in the root directory, so we specify None for the directory_name
        filename + '.json')

    file_service.create_file_from_path(
        'modellogs',
        None, # We want to create this blob in the root directory, so we specify None for the directory_name
        filename + '.pkl')