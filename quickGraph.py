import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce
from copy import deepcopy
from itertools import combinations


def getTrainTest(df, TrainSchool, TestSchool, asMatrix=True, includeFreshmen=True, yearMethod="none"):
    if not includeFreshmen:
        df = df[df["Year"] != 1]

    if yearMethod == "divide":
        df = df.astype({column: float for column in df.columns[5:]})
        for i in range(df.shape[0]):
            df.iloc[i,5:] /= df.iloc[i]["Year"]
    elif yearMethod == "proportion":
        df = df.astype({column: float for column in df.columns[5:]})
        for i in range(df.shape[0]):
            df.iloc[i,5:] /= df.iloc[i, 5:].sum()

    Train = df[df["School"] == TrainSchool]
    Test = df[df["School"] == TestSchool]

    X_train = Train.iloc[:, 5:]
    y_train = Train["Event"]

    X_test = Test.iloc[:, 5:]
    y_test = Test["Event"]

    if yearMethod == "feature":
        X_train.insert(0, "Year", Train["Year"])
        X_test.insert(0, "Year", Test["Year"])

    if asMatrix:
        return X_train.to_numpy(), y_train.values, X_test.to_numpy(), y_test.values
    else:
        return X_train, y_train, X_test, y_test


# make the graph
def makeGraph(data):
    # Filter the dataframe and make the graph
    G = nx.Graph()

    # Add athletes
    athletes = data.index
    for athlete in athletes:
        row = data.loc[athlete]
        G.add_node(
            athlete,
            type="athlete",
            Gender=row[0],
            School=row[1],
            Year=row[3],
            Event=row[4],
        )

    # Add events
    events = data.columns[5:]
    for event in events:
        G.add_node(
            event,
            type="event",
            uniqueAthletes=sum([1 if x != 0 else 0 for x in data[event]]),
            TimesDone=sum(data[event]),
        )

    # add edges
    for athlete in athletes:
        for event in events:
            if data.loc[athlete][event] != 0:
                G.add_edge(athlete, event, timesCompeted=data.loc[athlete][event])

    return G


# Output the graph
def drawGraph(G, labels=0):
    # This in inefficient but I like having it contained within the function
    def labelNode(node, labels):
        # No Labels
        if labels == 0:
            return ""

        # What to label athletes
        if "," in node:
            if labels == 2:
                last, first = node.split(", ")
                return first[0] + last[0]
            else:
                return ""
        # What to label events
        else:
            return node

    # Put the details in
    colors = [
        "Blue" if node[1]["type"] == "athlete" else "Red" for node in G.nodes(data=True)
    ]
    labels = {node: labelNode(node, labels) for node in G.nodes()}

    # Draw it
    nx.draw(G, labels=labels, node_color=colors)
    plt.show()


def setCheck(candidateEvents, uniqueMaximals):
    # If something in uniqueMaximals is a supserset of candidatEvents we don't
    #  candidatEvents
    # If something in uniqueMaximals is a subset of candidatEvents we add
    #   candidatEvents and remove it
    supersets = set()
    subsets = set()

    for athlete, events in uniqueMaximals.items():
        if events.issuperset(candidateEvents):
            supersets.add(athlete)
            break
        # Since issuperset includes equality we know gives only proper supersets
        elif events.issubset(candidateEvents):
            subsets.add(athlete)

    return supersets, subsets


def getUniqueMaximals(subsets):
    uniqueMaximals = {}
    for athlete, events in subsets.items():
        # people in uniqueMaximals who are subsets/supersets of events
        supersets, subsets = setCheck(events, uniqueMaximals)
        if not supersets:
            uniqueMaximals[athlete] = events
            # remove any subsets in uniqueMaximals from adding the new events
            for noLongerMaximal in subsets:
                del uniqueMaximals[noLongerMaximal]

    return uniqueMaximals


def subGraph(subsets):
    H = nx.Graph()

    for athlete in subsets.keys():
        H.add_node(athlete, type="athlete")

    for event in reduce(lambda x, y: x.union(y), subsets.values()):
        H.add_node(event, type="event")

    for athlete in subsets.keys():
        for event in subsets[athlete]:
            H.add_edge(athlete, event)

    return H


# taken from martinbroadhurst.com/greedy-set-cover-in-python.html
def set_cover(universe, subsets):
    """Find a family of subsets that covers the universal set"""
    elements = set(e for s in subsets for e in s)

    # Check the subsets cover the universe
    if elements.issubset(universe) and elements != universe:
        return None

    covered = set()
    cover = []

    # Greedily add the subsets with the most uncovered points
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s - covered))
        cover.append(subset)
        covered |= subset

    return cover


def removeMultiEvents(subsets, G):
    newSubsets = deepcopy(subsets)

    for athlete, events in subsets.items():
        if "Pent" in events:
            newSubsets[athlete] -= set(["HJ", "SP", "LJ", "60H", "800"])
        if "Hep" in events:
            if G.nodes(data=True)[athlete]["Gender"] == "F":
                newSubsets[athlete] -= set(["100H", "HJ", "SP", "200", "LJ", "JT", "800"])
            else:
                newSubsets[athlete] -= set(["1000", "60", "LJ", "SP", "HJ", "60H", "PV"])
        if "Dec" in events:
            newSubsets[athlete] -= set(["100", "LJ", "SP", "HJ", "400", "110H", "DT", "PV", "JT", "1500"])

    return newSubsets

"""def optimizeFurther(subsets, G):
    # find the degree 1 events, who does them, and those people's events
    lonelyEvents = [node for node, degree in G.degree() if degree == 1]
    H = G.copy()

    # find the people
    trivialPeople = set()
    for event in lonelyEvents:
        trivialPeople |= set(H[event])

    # find the events
    trivialEvents = set()
    for person in trivialPeople:
        trivialEvents |= set(H[person])

    # remove them
    for node in trivialPeople.union(trivialEvents):
        H.remove_node(node)

    return {node: set(H[node]) for node in H.nodes() if node in subsets.keys()}, trivialPeople, trivialEvents

### MAKES GREEDY WORSE
# Remove the 2-hop neighborhood of degree 1 vertices
    if False:
    reducedMaxAthletes, trivialPeople, trivialEvents = optimizeFurther(maxAthletes, H)
    maxAthletes = reducedMaxAthletes
    universe -= trivialEvents"""


if __name__ == "__main__":
    df = pd.read_csv("./fullDetails.csv", index_col=0)
    #X_train, y_train, X_test, y_test = getTrainTest(df, "RPI", "RIT", True)
    G = makeGraph(df[df.School == "RPI"])
    #drawGraph(G, 2)

    events = [node for node, stuff in dict(G.nodes(data=True)).items() if stuff["type"] == "event"]
    sorted([x for x in nx.eigenvector_centrality(G).items() if x[0] in events], key=lambda x: x[1], reverse=True)

    """
    # get sets for set cover problem
    # need minimal sets, handle multis, keep track of names
    universe = set(df.columns[5:])
    subsets = {
        node[0]: set(G[node[0]])
        for node in G.nodes(data=True)
        if node[1]["type"] == "athlete"
    }
    subsets = removeMultiEvents(subsets, G)

    # Initial pass
    maxAthletes = getUniqueMaximals(subsets)
    H = subGraph(maxAthletes)
    #drawGraph(H, 2)

    # Greedy algorithm
    greedy_cover = set_cover(universe, list(maxAthletes.values()))

    greedy_cover = {
        athlete: events
        for athlete, events in maxAthletes.items()
        if events in greedy_cover
    }

    J = subGraph(greedy_cover)
    #drawGraph(J, 2)

    # all combinations

    combinations_cover = {}
    #for comb in combinations(maxAthletes.values(), 11):
    #    events = reduce(lambda x, y: x.union(y), comb)
    #    if events == universe:
    #        combinations_cover = comb
    #        break

    #combinations_cover = {
    #    athlete: events
    #    for athlete, events in maxAthletes.items()
    #    if events in combinations_cover
    #}
    K = subGraph(combinations_cover)
    drawGraph(K, 2)
    """
