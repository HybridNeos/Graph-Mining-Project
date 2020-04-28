import requests
import re
from bs4 import BeautifulSoup
import pandas as pd


def simplifyEvent(text):
    temp = text.strip()
    if " " in temp:
        return temp[: temp.index(" ")]
    else:
        return temp

def RPIEventGroups(gender):
    # Get the html
    s = requests.Session()
    gender = "womens" if gender == "F" else "mens"
    r = s.get("https://rpiathletics.com/sports/{}-track-and-field/roster/2019-20".format(gender))
    soup = BeautifulSoup(r.text, features="lxml")

    # Parse stuff out
    firstNames = soup.findAll("span", class_="sidearm-roster-player-first-name")
    lastNames = soup.findAll("span", class_="sidearm-roster-player-last-name")
    events = soup.findAll(
        "div",
        class_="sidearm-list-card-details-item sidearm-roster-player-position-short",
    )

    return pd.DataFrame.from_dict(
        {
            last.get_text() + ", " + first.get_text(): simplifyEvent(event.get_text())
            for first, last, event in zip(firstNames, lastNames, events)
        },
        orient="index",
        columns=["Event"]
    )

EventGroups = pd.concat([EventGroups("M"), RPIEventGroups("F")])
