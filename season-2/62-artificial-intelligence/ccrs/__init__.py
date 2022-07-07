"""
Cannlytics Metrc Client Initialization | Cannlytics
Copyright (c) 2021-2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 11/6/2021
Updated: 11/6/2021
"""
from typing import Any, Optional
from .client import Metrc


def initialize_metrc(
        vendor_api_key: str,
        user_api_key: str,
        primary_license: Optional[str] = '',
        state: Optional[str] = 'ca',
        client_class: Any = Metrc,
) -> Metrc:
    """This is a shortcut function which instantiates a Metrc
    client using a user API key and the vendor API key.
    Args:
        vendor_api_key (str): The vendor's API key.
        user_api_key (str): The user's API key.
        primary_license (str): An optional primary license to use if no license is specified.
        state (str): The state of the traceability system, `ca` by default.
        client_class: By default :class:`cannlytics.metrc.client.Client` is used.
    Returns:
        (Metrc): Returns an instance of the Metrc client.
    """
    return client_class(
        vendor_api_key,
        user_api_key,
        primary_license=primary_license,
        state=state
    )
