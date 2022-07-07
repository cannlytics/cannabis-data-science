"""
CCRS Models | Cannlytics
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/10/2022
Updated: 4/10/2022
License: <https://github.com/cannlytics/cannlytics-engine/blob/main/LICENSE>

This module contains common CCRS models.
"""

# Internal imports.
from cannlytics.firebase import get_document, update_document
from cannlytics.utils.utils import (
    camel_to_snake,
    clean_dictionary,
    clean_nested_dictionary,
    get_timestamp,
    remove_dict_fields,
    remove_dict_nulls,
    snake_to_camel,
    update_dict,
)


class Model(object):
    """Base class for all Metrc models."""

    def __init__(
            self,
            client,
            context,
            license_number='',
            function=camel_to_snake
    ):
        """Initialize the model, setting keys as properties."""
        self.client = client
        self._license = license_number
        properties = clean_nested_dictionary(context, function)
        for key in properties:
            self.__dict__[key] = properties[key]

    def __getattr__(self, key):
        return self.__dict__[key]

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    @property
    def uid(self):
        """The model's unique ID."""
        return self.__dict__.get('id')

    @classmethod
    def from_dict(cls, client, json):
        """Initiate a class instance from a dictionary."""
        obj = cls(client, json)
        try:
            obj.create()
        except KeyError:
            pass
        return obj

    @classmethod
    def from_fb(cls, client, ref):
        """Initialize a class from Firebase data.
        Args:
            client (Client): A Metrc client instance.
            ref (str): The reference to the document in Firestore.
        Returns:
            (Model): A Metrc model.
        """
        data = get_document(ref)
        obj = cls(client, data)
        return obj

    def to_dict(self):
        """Returns the model's properties as a dictionary."""
        data = vars(self).copy()
        [data.pop(x, None) for x in ['_license', 'client', '__class__']]
        return data

    def to_fb(self, ref='', col=''):
        """Upload the model's properties as a dictionary to Firestore.
        Args:
            ref (str): The Firestore document reference.
            col (str): A Firestore collection, with the UID as document ID.
        """
        data = vars(self).copy()
        [data.pop(x, None) for x in ['_license', 'client']]
        if col:
            update_document(f'{col}/{self.uid}', data)
        else:
            update_document(ref, data)



# TODO: Areas

# TODO: Contacts

# TODO: Integrators

# TODO: Inventory

# TODO: Inventory Adjustments

# TODO: Inventory Plant Transfers

# TODO: Lab Results

# TODO: Licensees

# TODO: Plants

# TODO: Plant destructions

# TODO: Products

# TODO: Sale Headers

# TODO: Sale Details

# TODO: Strains

# TODO: Transfers (hard)
