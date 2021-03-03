#-----------------------------------------------------------------------------
# Copyright (c) 2021, Cannlytics, and Cannlytics Contributors.
# All rights reserved.
#
# License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
# This is free software: you are free to change and redistribute it.
# There is NO WARRANTY, to the extent permitted by law.
#-----------------------------------------------------------------------------
""" Provide the ``Traceability`` class.

Traceability instances are factories for making requests to state traceability
systems, such as Leaf Data Systems and Metrc.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# Globals and constants
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# Dev API
#-----------------------------------------------------------------------------

class Traceability:
    """ Traceability is a factory for state traceability requests.
    """


    def __init__(self, *handlers, **kwargs):
        """ Traceability factory.

        Args:
            handlers (seq[Handler]): List of handlers to call.

        Keyword Args:
            metadata (dict): arbitrary user-supplied JSON data to make available
                with the application.
        """
        metadata = kwargs.pop('metadata', None)
        if kwargs:
            raise TypeError("Invalid keyword argument: %s" %
                list(kwargs.keys())[0])
        self._static_path = None
        self._handlers = []
        self._metadata = metadata
        for h in handlers:
            self.add(h)
        
    # Properties --------------------------------------------------------------
    
    @property
    def handlers(self):
        """ The ordered list of handlers this Traceability is configured with.
        """
        return tuple(self._handlers)
    
    # Public methods ----------------------------------------------------------

    def add(self, handler):
        """ Add a handler to the pipeline used to initialize new documents.
        Args:
            handler (Handler) : a handler for this Application to use to
                process Documents
        """
        self._handlers.append(handler)

        # make sure there is at most one static path
        static_paths = {h.static_path() for h in self.handlers}
        static_paths.discard(None)
        if len(static_paths) > 1:
            raise RuntimeError("More than one static path requested for app: %r" % list(static_paths))
        elif len(static_paths) == 1:
            self._static_path = static_paths.pop()
        else:
            self._static_path = None

    