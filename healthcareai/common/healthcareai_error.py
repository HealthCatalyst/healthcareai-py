class HealthcareAIError(Exception):
    """ This is the error that should be thrown to help communicate problems to users in a nice way. """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)

    pass
