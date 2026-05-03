# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# This module contains Dysmalpy custom exceptions

class NoordermeerFlattenerError(Exception):
    def __init__(self, message):            
        super().__init__(message)