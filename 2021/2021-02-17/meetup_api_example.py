# -*- coding: utf-8 -*-
"""
Meetup API | Cannabis Data Science

Author: Keegan Skeate
Created: Mon Feb 15 09:21:32 2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:
    Get members and events for the Cannabis Data Science Meetup Group.

Resources:
    https://asp.net-hacker.rocks/2017/03/13/integrate-meetup-events-on-your-website.html

"""
import requests


def get_events(meetup):
    """Get meet up events."""
    base = f"https://api.meetup.com/{meetup}/events"
    response = requests.get(base)
    return response.json()


def get_members(meetup):
    """Get meet up members."""
    base = f"https://api.meetup.com/{meetup}/members"
    response = requests.get(base)
    return response.json()


if __name__ == "__main__":
    
    meetup = "cannabis-data-science"
    
    events = get_events("cannabis-data-science")
    print(events[0])
    
    members = get_members("cannabis-data-science")
    print(members[0])


