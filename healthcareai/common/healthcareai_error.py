class HealthcareAIError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)

    pass


if __name__ == "__main__":
    pass
