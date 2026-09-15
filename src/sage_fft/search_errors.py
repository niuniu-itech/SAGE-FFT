# Author: even
"""Recoverable candidate failures; shared build and transport errors stay fatal."""


class CandidateRejected(Exception):
    def __init__(self, message, *, phase, measurement=None):
        super().__init__(message)
        self.phase = phase
        self.measurement = measurement


class CandidateLoweringError(CandidateRejected):
    def __init__(self, message):
        super().__init__(message, phase="lowering")


class CandidateCorrectnessError(CandidateRejected):
    def __init__(self, message, measurement):
        super().__init__(message, phase="correctness", measurement=measurement)
