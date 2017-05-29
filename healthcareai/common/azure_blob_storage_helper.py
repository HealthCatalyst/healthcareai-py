import pickle

from azure.storage.blob import BlockBlobService
from azure.common import AzureMissingResourceHttpError


class AzureBlobStorageHelper:
    """Azure Blob Storage Helper

    This class helps you store blobs on Azure's Blob Storage Service.

    Note you do need to have the azure-storage python package installed. It is intentionally not included here to keep the
    dependencies of healthcareai light.

    After instantiating the class with your Azure account name and key you can easily upload text blobs and objects
    (that this class automatically pickles for you).

    You can also create a new container (if it doesn't already exist).

    Examples:
        ```
        my_azure = AzureHelper('fakeyname123', 'fakeyfakefakefakekey12312312')
        my_azure.create_container('my_new_container')
        my_azure.save_text_blob('text goes here...', 'my_filename.txt', 'my_new_container')

        stuff = {'model_type': 'random_forest', 'weights': [2, 5, 1, 3, 6], 'hyper_parameters': {'a': 33.3, 'b': 42.2}}
        my_azure.save_object_as_pickle(stuff, 'stuff.pkl', 'my_new_container')

        ```

    """

    def __init__(self, account_name, account_key):
        """
        Instantiate with your Azure account name and account key
        :param account_name: account name
        :param account_key: account key
        """
        self._account_name = account_name
        self._account_key = account_key
        self._connection = self._create_azure_connection()

    def _create_azure_connection(self):
        """Returns and instance of BlockBlobService"""
        return BlockBlobService(account_name=self._account_name, account_key=self._account_key)

    def save_text_blob(self, blob, blob_name, container):
        """
        Saves a blob of text to azure
        :param blob: the blob of text
        :param blob_name: the name of the file on the container
        :param container: the name of the container
        :return:
        """
        return self._connection.create_blob_from_text(container_name=container, blob_name=blob_name, text=blob)

    def save_object_as_pickle(self, object_to_pickle, blob_name, container):
        """
        Save an object as a pickle file to azure
        :param object_to_pickle: the object you want pickled 'n shipped
        :param blob_name: the name of the file on the container
        :param container: the name of the container
        :return:
        """
        # Note the explicit use of dumps rather than dump
        blob = pickle.dumps(object_to_pickle)
        return self._connection.create_blob_from_bytes(container_name=container, blob_name=blob_name, blob=blob)

    def create_container(self, container_name):
        """
        Creates a new container on azure if it does not exist.
        :param container_name: the name of the container
        """
        try:
            return self._connection.create_container(container_name)
        except AzureMissingResourceHttpError as ae:
            raise (AzureBlobStorageHelperError('The specified container does not exist.'))


class AzureBlobStorageHelperError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


if __name__ == "__main__":
    pass
